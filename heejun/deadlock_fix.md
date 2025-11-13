# SOAR Pipeline Deadlock 문제 해결

## 문제 진단
날짜: 2025-11-13

### 증상
- 로그가 06:36:57에 멈춤
- vLLM 서버는 06:37:03에 모든 요청 처리 완료 (Running: 0 reqs)
- 프로세스는 실행 중이지만 progress.json 생성 안됨
- 8개 task가 동시에 시작됨

### 근본 원인
1. **과도한 타임아웃**: AsyncOpenAI 클라이언트의 timeout=2500초 (41분 40초)
2. **vLLM 서버 과부하**: 8개 task가 동시에 시작되면서 서버 응답 지연
3. **네트워크 데드락**: 서버는 처리 완료했지만 클라이언트는 계속 대기

## 해결 방법

### 1. 타임아웃 설정 수정
**파일**: `/data/hjkim/soar2cot/src/llms/structured.py`
**라인**: 147

```python
# 수정 전
local_vllm_client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=os.environ.get("LOCAL_VLLM_URL", "http://localhost:8000/v1"),
    timeout=2500,  # 너무 김!
    max_retries=2,
)

# 수정 후 (권장)
local_vllm_client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=os.environ.get("LOCAL_VLLM_URL", "http://localhost:8000/v1"),
    timeout=60,  # 60초로 단축
    max_retries=3,  # 재시도 횟수 증가
)
```

### 2. 병렬 처리 시작 지연 증가
**파일**: `/data/hjkim/soar2cot/src/run.py`
**라인**: 1344

```python
# 수정 전
futures.append(process_single_task(task_id, task_row, sleep_for=i * 2))

# 수정 후 (권장)
futures.append(process_single_task(task_id, task_row, sleep_for=i * 5))  # 5초 간격으로 증가
```

### 3. 동시 처리 task 수 조정 (선택적)
**파일**: `/data/hjkim/soar2cot/src/configs/oss_configs.py`
**라인**: 44

```python
# 현재
max_concurrent_tasks=8,

# 필요시 줄이기
max_concurrent_tasks=4,  # vLLM 서버 부하 감소
```

## 환경 변수 설정 (선택적)

```bash
# MAX_CONCURRENCY 환경 변수 조정
export MAX_CONCURRENCY=50  # 기본값 100에서 50으로 감소
```

## 되돌리기 방법

### Git을 사용한 되돌리기
```bash
cd /data/hjkim/soar2cot
git diff src/llms/structured.py  # 변경 사항 확인
git checkout -- src/llms/structured.py  # 되돌리기
```

### 수동 되돌리기
1. timeout을 2500으로 복원
2. sleep_for를 i * 2로 복원
3. max_concurrent_tasks를 8로 복원

## 테스트 방법

1. **단일 task 테스트**:
```bash
python -m src.run --task_ids 007bbfb7 --num_tasks 1
```

2. **병렬 처리 테스트**:
```bash
python test/test_deadlock.py
```

3. **로그 모니터링**:
```bash
tail -f logs/arc_*.log
```

## 모니터링 포인트

1. vLLM 서버 상태:
```bash
tail -f /data/arclang/logs/vllm_server_*.log | grep "Running:"
```

2. 프로세스 상태:
```bash
ps aux | grep "python -m src.run"
```

3. progress.json 생성 여부:
```bash
watch -n 5 ls -lh data/progress.json
```

## 예상 효과

- 타임아웃 발생 시 60초 내에 재시도
- vLLM 서버 부하 감소로 응답 속도 개선
- 데드락 방지로 안정적인 처리

## 주의 사항

- GPU 0,1,2,3만 사용하도록 설정 유지
- vLLM 서버는 tensor_parallel=4로 실행 중
- 변경 후 반드시 테스트 실행