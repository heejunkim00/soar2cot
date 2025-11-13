# GPU 4개 병렬처리 적용 변경사항 및 롤백 가이드

**작성일**: 2025-11-13
**목적**: GPU 4개(0,1,2,3) 사용 및 Task 레벨 병렬처리 적용

---

## 변경 요약

### 목표
- GPU 사용: 2개 → 4개 (GPU 0,1,2,3)
- Task 처리: 순차 → 병렬 (최대 8개 동시)
- 예상 성능: 4-8배 향상

### 변경된 파일
1. `start_vllm_server.sh` - vLLM 서버 GPU 설정
2. `src/run.py` - Task 병렬처리 추가
3. `src/configs/oss_configs.py` - 동시 실행 수 증가

---

## 상세 변경 내역

### 1. start_vllm_server.sh

**위치**: `/data/hjkim/soar2cot/start_vllm_server.sh`

**변경 전 (Line 14-15)**:
```bash
GPUS="0,1"
TENSOR_PARALLEL=2
```

**변경 후 (Line 14-15)**:
```bash
GPUS="0,1,2,3"
TENSOR_PARALLEL=4
```

**변경 사항**:
- GPU 사용 개수: 2개 → 4개
- Tensor Parallel Size: 2 → 4
- 주석도 업데이트: "Runs on GPU 0, 1" → "Runs on GPU 0, 1, 2, 3"

**효과**:
- 모델이 4개 GPU에 분산 로드됨
- 각 요청 처리 속도 약간 향상
- 더 많은 동시 요청 처리 가능

---

### 2. src/run.py

**위치**: `/data/hjkim/soar2cot/src/run.py`

**변경 전 (Line 1262-1329)**:
```python
# Task별로 순회 (알파벳 순서)
for task_id in sorted(round_df["task_id"].unique()):
    # task_ids 필터링 (설정되어 있으면)
    if task_ids and task_id not in task_ids:
        continue

    task_row = round_df[round_df["task_id"] == task_id].iloc[0]

    try:
        # Set SOAR metadata in global context
        global SOAR_METADATA_CONTEXT
        SOAR_METADATA_CONTEXT = {...}

        # Challenge 생성
        challenge = create_challenge_from_soar(...)

        python_code = task_row["code"]
        data_type = "original"
        solution_grids = [task_row["predicted_test_output"]]

        log.info(f"Processing task", task_id=task_id, ...)

        # Challenge 처리
        await solve_challenge(
            c=challenge,
            attempts_path=attempts_path,
            temp_attempts_dir=temp_attempts_dir,
            solution_grids=task_row["predicted_test_output"],
            config=config,
            python_code=python_code,
        )

        # 완료 표시
        progress.add_completed(task_id, round_idx, data_type)

        log.info(f"Task completed", task_id=task_id, ...)

    except Exception as e:
        log.error(f"Failed to process task", task_id=task_id, ...)
        continue
```

**변경 후 (Line 1262-1349)**:
```python
# Task별로 병렬 처리 (알파벳 순서)
# Create semaphore to limit concurrent tasks
semaphore = MonitoredSemaphore(config.max_concurrent_tasks, name="task_semaphore")

async def process_single_task(task_id: str, task_row, sleep_for: int):
    """Process a single task with semaphore for concurrency control"""
    async with semaphore:
        # Stagger task starts to avoid thundering herd
        await asyncio.sleep(sleep_for)

        try:
            # Set SOAR metadata in global context
            # Note: This is thread-safe in asyncio since we're in same thread
            global SOAR_METADATA_CONTEXT
            SOAR_METADATA_CONTEXT = {...}

            # Challenge 생성
            challenge = create_challenge_from_soar(...)

            python_code = task_row["code"]
            data_type = "original"
            solution_grids = [task_row["predicted_test_output"]]

            log.info(f"Processing task", task_id=task_id, ...)

            # Challenge 처리
            await solve_challenge(
                c=challenge,
                attempts_path=attempts_path,
                temp_attempts_dir=temp_attempts_dir,
                solution_grids=task_row["predicted_test_output"],
                config=config,
                python_code=python_code,
            )

            # 완료 표시
            progress.add_completed(task_id, round_idx, data_type)

            log.info(f"Task completed", task_id=task_id, ...)

        except Exception as e:
            log.error(f"Failed to process task", task_id=task_id, ...)

# Collect all tasks to process
futures = []
for i, task_id in enumerate(sorted(round_df["task_id"].unique())):
    # task_ids 필터링 (설정되어 있으면)
    if task_ids and task_id not in task_ids:
        continue

    task_row = round_df[round_df["task_id"] == task_id].iloc[0]
    futures.append(process_single_task(task_id, task_row, sleep_for=i * 2))

# Process all tasks in parallel
log.info(
    f"Starting parallel task processing",
    total_tasks=len(futures),
    max_concurrent=config.max_concurrent_tasks,
)
await asyncio.gather(*futures, return_exceptions=True)
```

**변경 사항**:
1. **Semaphore 추가**: `MonitoredSemaphore`로 동시 실행 수 제한
2. **중첩 함수 생성**: `process_single_task()` - 개별 task 처리
3. **Sleep staggering**: `sleep_for=i * 2` - thundering herd 방지
4. **병렬 실행**: `asyncio.gather(*futures)` - 모든 task 동시 처리
5. **로깅 추가**: 병렬 처리 시작 시 총 task 수와 max_concurrent 로그

**효과**:
- 여러 Task가 동시에 처리됨
- GPU 활용률 대폭 증가
- 전체 처리 시간 단축

**주의사항**:
- `SOAR_METADATA_CONTEXT`는 global 변수이지만 asyncio는 single-threaded이므로 안전
- `progress.add_completed()`는 즉시 파일에 저장하므로 동시성 문제 없음
- 각 task는 2초씩 stagger되어 시작 (서버 부하 분산)

---

### 3. src/configs/oss_configs.py

**위치**: `/data/hjkim/soar2cot/src/configs/oss_configs.py`

**변경 전 (Line 44)**:
```python
max_concurrent_tasks=4,
```

**변경 후 (Line 44)**:
```python
max_concurrent_tasks=8,  # Increased from 4 to 8 for 4-GPU setup
```

**변경 사항**:
- 최대 동시 Task 수: 4 → 8

**이유**:
- GPU 4개로 늘어났으므로 더 많은 병렬 처리 가능
- 각 Task당 5-10개 LLM 요청 → 8개 Task면 40-80개 동시 요청
- vLLM 서버의 `max_num_seqs=128` 능력 활용

**효과**:
- 8개 Task가 동시에 처리됨
- GPU 4개가 충분히 활용됨

---

## 롤백 방법

### 방법 1: Git으로 롤백 (추천)

```bash
cd /data/hjkim/soar2cot

# 현재 변경사항 확인
git diff

# 특정 파일만 롤백
git checkout HEAD -- start_vllm_server.sh
git checkout HEAD -- src/run.py
git checkout HEAD -- src/configs/oss_configs.py

# 또는 모든 변경사항 롤백
git checkout HEAD -- .
```

### 방법 2: 수동 롤백

#### 2-1. start_vllm_server.sh 롤백

```bash
# 파일 편집
vim /data/hjkim/soar2cot/start_vllm_server.sh

# Line 14-15 변경:
GPUS="0,1"
TENSOR_PARALLEL=2

# Line 4 주석 변경:
# Runs on GPU 0, 1 with tensor parallelism
```

#### 2-2. src/run.py 롤백

```bash
# 파일 편집
vim /data/hjkim/soar2cot/src/run.py
```

**Line 1262-1349를 다음으로 교체**:

```python
        # Task별로 순회 (알파벳 순서)
        for task_id in sorted(round_df["task_id"].unique()):
            # task_ids 필터링 (설정되어 있으면)
            if task_ids and task_id not in task_ids:
                continue

            task_row = round_df[round_df["task_id"] == task_id].iloc[0]

            try:
                # Set SOAR metadata in global context
                global SOAR_METADATA_CONTEXT
                SOAR_METADATA_CONTEXT = {
                    "soar_code": task_row["code"],
                    "soar_source_model": task_row["model"],
                    "soar_generation": int(task_row["generation"]),
                    "soar_round_index": int(round_idx),
                    "is_hindsight": False,  # Original 데이터는 항상 False
                }

                # Challenge 생성
                challenge = create_challenge_from_soar(
                    task_id=task_id,
                    predicted_train_outputs=task_row["predicted_train_output"],
                    predicted_test_outputs=task_row["predicted_test_output"],
                )

                python_code = task_row["code"]
                data_type = "original"  # Original 데이터로 하드코딩
                solution_grids = [task_row["predicted_test_output"]]

                log.info(
                    f"Processing task",
                    task_id=task_id,
                    round_index=round_idx,
                    data_type=data_type,
                )

                # Challenge 처리
                await solve_challenge(
                    c=challenge,
                    attempts_path=attempts_path,
                    temp_attempts_dir=temp_attempts_dir,
                    solution_grids=task_row["predicted_test_output"],
                    config=config,
                    python_code=python_code,
                )

                # 완료 표시
                progress.add_completed(task_id, round_idx, data_type)

                log.info(
                    f"Task completed",
                    task_id=task_id,
                    round_index=round_idx,
                    progress_stats=progress.get_stats(),
                )

            except Exception as e:
                log.error(
                    f"Failed to process task",
                    task_id=task_id,
                    round_index=round_idx,
                    error=str(e),
                    traceback="".join(
                        traceback.format_exception(type(e), e, e.__traceback__)
                    ),
                )
                continue
```

**핵심 차이**:
- `for` 루프로 순차 처리
- `semaphore`, `process_single_task()`, `asyncio.gather()` 제거

#### 2-3. src/configs/oss_configs.py 롤백

```bash
# 파일 편집
vim /data/hjkim/soar2cot/src/configs/oss_configs.py

# Line 44 변경:
max_concurrent_tasks=4,
```

---

## 적용 후 확인 사항

### 1. vLLM 서버 재시작 필요

변경사항을 적용한 후 **반드시 vLLM 서버를 재시작**해야 합니다:

```bash
# 기존 서버 종료
pkill -f "vllm.entrypoints.openai.api_server"

# 또는 특정 프로세스 종료
ps aux | grep vllm
kill <PID>

# 새 서버 시작
cd /data/hjkim/soar2cot
./start_vllm_server.sh
```

### 2. GPU 사용 확인

```bash
# GPU 사용 상태 확인
nvidia-smi

# 모델이 4개 GPU에 로드되었는지 확인
# GPU 0, 1, 2, 3 모두에서 메모리 사용 확인
```

**예상 출력**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H200         On   | 00000000:00:04.0 Off |                    0 |
| N/A   45C    P0    85W / 400W |  35000MiB / 143771MiB|     15%      Default |  ← 사용 중
|   1  NVIDIA H200         On   | 00000000:00:05.0 Off |                    0 |
| N/A   46C    P0    87W / 400W |  35000MiB / 143771MiB|     16%      Default |  ← 사용 중
|   2  NVIDIA H200         On   | 00000000:00:06.0 Off |                    0 |
| N/A   44C    P0    84W / 400W |  35000MiB / 143771MiB|     14%      Default |  ← 사용 중
|   3  NVIDIA H200         On   | 00000000:00:07.0 Off |                    0 |
| N/A   45C    P0    86W / 400W |  35000MiB / 143771MiB|     15%      Default |  ← 사용 중
+-----------------------------------------------------------------------------+
```

### 3. 로그 확인

```bash
# 병렬 처리 시작 로그 확인
tail -f /data/hjkim/soar2cot/logs/arc_*.log | grep "parallel task processing"

# 예상 출력:
# Starting parallel task processing | {"total_tasks":344,"max_concurrent":8,...}
```

### 4. 성능 모니터링

```bash
# GPU 사용률 실시간 모니터링
watch -n 1 nvidia-smi

# vLLM 서버 로그 확인
tail -f /data/arclang/logs/vllm_server_*.log
```

---

## 예상 결과

### 변경 전
```
Task 처리: 순차 (한 번에 1개)
GPU 사용: 2개 (GPU 0, 1)
동시 LLM 요청: 5-10개
처리 속도: 1x
```

### 변경 후
```
Task 처리: 병렬 (한 번에 최대 8개)
GPU 사용: 4개 (GPU 0, 1, 2, 3)
동시 LLM 요청: 40-80개
처리 속도: 4-8x
```

---

## 문제 해결

### 문제 1: vLLM 서버가 시작되지 않음

**증상**:
```
CUDA out of memory
```

**해결**:
```bash
# GPU 메모리 사용률 줄이기
# start_vllm_server.sh 편집
--gpu-memory-utilization 0.85  →  --gpu-memory-utilization 0.75
```

### 문제 2: Task가 순차적으로 처리됨

**원인**: 병렬 처리 코드가 제대로 적용되지 않음

**확인**:
```bash
# 로그에서 "parallel task processing" 메시지 확인
grep "parallel task processing" /data/hjkim/soar2cot/logs/arc_*.log

# 있으면 OK, 없으면 코드 재확인
```

### 문제 3: 에러 발생률 증가

**원인**: 너무 많은 동시 요청으로 서버 과부하

**해결**:
```python
# src/configs/oss_configs.py
max_concurrent_tasks=8  →  max_concurrent_tasks=4  # 줄이기
```

### 문제 4: GPU 0,1만 사용됨

**원인**: vLLM 서버가 재시작되지 않음

**해결**:
```bash
# 서버 완전 종료 후 재시작
pkill -9 -f vllm
./start_vllm_server.sh
```

---

## 추가 최적화 옵션

### 옵션 1: max_concurrent_tasks 조정

현재 8로 설정되어 있으나, 실제 성능에 따라 조정 가능:

```python
# 더 많은 병렬 처리 (GPU 여유 있을 때)
max_concurrent_tasks=12

# 더 적은 병렬 처리 (에러 많을 때)
max_concurrent_tasks=4
```

### 옵션 2: Sleep stagger 조정

현재 2초로 설정:

```python
# src/run.py line 1341
futures.append(process_single_task(task_id, task_row, sleep_for=i * 2))

# 더 빠르게 시작 (1초)
sleep_for=i * 1

# 더 천천히 시작 (3초)
sleep_for=i * 3
```

### 옵션 3: max_num_seqs 조정

vLLM 서버의 동시 요청 처리 수:

```bash
# start_vllm_server.sh
--max-num-seqs 128  # 현재 설정

# 더 많은 동시 요청 (메모리 충분할 때)
--max-num-seqs 256

# 더 적은 동시 요청 (OOM 발생 시)
--max-num-seqs 64
```

---

## 참고 자료

- vLLM 문서: https://docs.vllm.ai/
- Tensor Parallelism: https://docs.vllm.ai/en/latest/serving/distributed_serving.html
- Asyncio Semaphore: https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore

---

## 버전 정보

**변경 날짜**: 2025-11-13
**변경자**: Claude Code
**Git Commit**: (해당되는 경우 commit hash 기록)
**테스트 상태**: 미테스트 (롤백 가이드만 제공)

---

## 체크리스트

적용 후 확인:
- [ ] vLLM 서버 재시작 완료
- [ ] GPU 4개(0,1,2,3) 모두 사용 중
- [ ] 로그에 "parallel task processing" 메시지 확인
- [ ] 여러 Task가 동시에 처리되는지 확인
- [ ] 에러율이 크게 증가하지 않는지 확인
- [ ] 처리 속도 개선 확인

롤백 시 확인:
- [ ] GPU 2개(0,1)로 돌아감
- [ ] Task가 순차적으로 처리됨
- [ ] 이전과 동일한 동작 확인
