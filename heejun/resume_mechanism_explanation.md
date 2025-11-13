# 파이프라인 중단 및 재시작 메커니즘 상세 설명

## 개요

파이프라인이 **중단되어도 처음부터 다시 시작하지 않고**, 이전에 완료한 작업은 건너뛰고 **중단된 지점부터 재개**됩니다.

---

## 핵심 구성 요소

### 1. Progress Tracker (`src/progress_tracker.py`)

**역할**: 완료된 작업을 추적하는 작은 JSON 파일 관리

**저장 위치**: `/data/hjkim/soar2cot/data/progress.json`

**구조**:
```json
{
  "completed": [
    {
      "task_id": "007bbfb7",
      "round_index": 0,
      "data_type": "original"
    },
    {
      "task_id": "00d62c1b",
      "round_index": 0,
      "data_type": "original"
    }
  ],
  "total_completed": 2,
  "last_saved_to_parquet": 0
}
```

**주요 메서드**:
```python
class ProgressTracker:
    def get_completed_set(self) -> Set[Tuple[str, int, str]]:
        """완료된 (task_id, round_index, data_type) 조합을 Set으로 반환"""

    def add_completed(self, task_id: str, round_index: int, data_type: str):
        """작업 완료 시 즉시 저장"""
```

---

## 재시작 프로세스 (단계별)

### Step 1: 파이프라인 시작

```python
# src/run.py line 1200-1201
progress = ProgressTracker()
log.info("Progress tracker initialized", stats=progress.get_stats())
```

**동작**:
- `progress.json` 파일이 존재하면 → 로드
- 파일이 없으면 → 빈 상태로 시작

**로그 출력**:
```
Progress tracker initialized stats={'total_completed': 2, ...}
# 또는
No progress file found, starting fresh
```

---

### Step 2: SOAR 데이터 로드 및 필터링

```python
# src/run.py line 1205-1208
soar_df = load_soar_data_labeled(
    path=Path("/data/hjkim/soar2cot/data/soar_arc_train_5M_original.parquet"),
    progress_tracker=progress,  # ← progress tracker 전달
)
```

#### 내부 동작 (`src/soar_loader.py` line 266-285):

```python
if progress_tracker:
    completed_set = progress_tracker.get_completed_set()
    # completed_set = {('007bbfb7', 0, 'original'), ('00d62c1b', 0, 'original')}

    if completed_set:
        # 이미 완료된 샘플을 DataFrame에서 제거
        mask = df.apply(
            lambda row: (
                row["task_id"],
                row["round_index"],
                row["data_type"],
            )
            not in completed_set,  # ← 완료되지 않은 것만 True
            axis=1,
        )
        df = df[mask].copy()  # 필터링된 DataFrame

        log.info(
            "Filtered out completed samples",
            remaining=len(df),
            completed=len(completed_set),
        )
```

**결과**:
- 전체 SOAR 데이터: 5,000,000개
- 이미 완료: 2개
- 남은 작업: 4,999,998개 ← 이것만 처리

---

### Step 3: 작업 처리

```python
# src/run.py line 1226-1298
for round_idx in range(max_round + 1):
    round_df = soar_df[soar_df["round_index"] == round_idx]

    for task_id in sorted(round_df["task_id"].unique()):
        # Challenge 처리
        await solve_challenge(
            c=challenge,
            solution_grids=task_row["predicted_test_output"],
            config=config,
            python_code=python_code,
        )

        # ✅ 완료 즉시 저장
        progress.add_completed(task_id, round_idx, data_type)
```

#### `progress.add_completed()` 동작 (`src/progress_tracker.py` line 63-91):

```python
def add_completed(self, task_id: str, round_index: int, data_type: str):
    # 1. completed 리스트에 추가
    self.data["completed"].append({
        "task_id": task_id,
        "round_index": round_index,
        "data_type": data_type,
    })

    # 2. 총 개수 업데이트
    self.data["total_completed"] = len(self.data["completed"])

    # 3. 즉시 파일에 저장 (중요!)
    self.save()  # ← progress.json에 즉시 기록

    log.debug("Sample marked as completed", task_id=task_id, ...)
```

**중요**: 각 작업이 완료될 때마다 **즉시 progress.json에 저장**됩니다!

---

## 중단 및 재시작 시나리오

### 시나리오 1: 정상 중단 (Ctrl+C)

```bash
# 파이프라인 실행 중
$ python -m src.run

# 처리 중...
Processing task task_id=007bbfb7 round_index=0
Task completed task_id=007bbfb7  # ← progress.json 저장됨
Processing task task_id=00d62c1b round_index=0
Task completed task_id=00d62c1b  # ← progress.json 저장됨
Processing task task_id=05269061 round_index=0
^C  # ← Ctrl+C로 중단

# progress.json 상태:
{
  "completed": [
    {"task_id": "007bbfb7", "round_index": 0, "data_type": "original"},
    {"task_id": "00d62c1b", "round_index": 0, "data_type": "original"}
  ],
  "total_completed": 2
}
```

**재시작**:
```bash
# 다시 실행
$ python -m src.run

# 로그:
Progress tracker initialized stats={'total_completed': 2}
Loading original SOAR data
Filtered out completed samples remaining=4999998 completed=2

# 결과: 007bbfb7, 00d62c1b는 건너뛰고 05269061부터 재개!
Processing task task_id=05269061 round_index=0
```

---

### 시나리오 2: 비정상 종료 (서버 다운, 메모리 부족 등)

```bash
# 파이프라인 실행 중
Processing task task_id=007bbfb7 round_index=0
Task completed task_id=007bbfb7  # ← progress.json 저장됨
Processing task task_id=00d62c1b round_index=0
[서버 갑자기 다운]  # ← 00d62c1b는 미완료

# progress.json 상태:
{
  "completed": [
    {"task_id": "007bbfb7", "round_index": 0, "data_type": "original"}
  ],
  "total_completed": 1
}
```

**재시작**:
```bash
# 서버 복구 후 재실행
$ python -m src.run

# 결과: 007bbfb7은 건너뛰고, 00d62c1b부터 재개
Processing task task_id=00d62c1b round_index=0
```

**손실**: 00d62c1b 작업은 **처음부터 다시** 시작 (완료 직전에 중단되었더라도)

---

### 시나리오 3: 에러로 인한 중단

```python
# 특정 task에서 에러 발생
Processing task task_id=abc123 round_index=0
[ERROR] Exception in solve_challenge: ...

# progress.json: abc123는 저장 안 됨 (add_completed 호출 전 에러)
# 다음 실행 시 abc123 다시 시도
```

---

## 재시작 시 데이터 처리 흐름

```
1. 파이프라인 시작
   ↓
2. ProgressTracker 초기화
   - progress.json 로드 (있으면)
   - completed_set = {('007bbfb7', 0, 'original'), ...}
   ↓
3. SOAR 데이터 로드
   - 전체 5M 샘플 로드
   ↓
4. 완료된 샘플 필터링
   - completed_set에 있는 샘플 제거
   - 남은 샘플만 DataFrame에 유지
   ↓
5. Round별 순회
   - Round 0, 1, 2, ...
   ↓
6. Task별 처리
   - 각 task 완료 시마다 progress.json 업데이트
   ↓
7. 중단되면
   - progress.json에 완료된 것까지만 기록됨
   ↓
8. 재시작
   - Step 2로 돌아가서 반복
   - 이전에 완료된 작업은 자동으로 건너뜀
```

---

## 핵심 메커니즘

### 1. **즉시 저장 (Immediate Save)**

```python
# src/progress_tracker.py line 82-83
# 작업 완료 즉시 파일에 기록
self.save()
```

**장점**:
- 중단되어도 최소한의 손실
- 마지막 완료된 작업까지는 저장됨

**단점**:
- 매번 파일 I/O 발생
- 하지만 progress.json은 작은 파일이므로 빠름

### 2. **Set 기반 필터링**

```python
# src/soar_loader.py line 267
completed_set = progress_tracker.get_completed_set()
# Set: O(1) lookup time

# line 275-276
(row["task_id"], row["round_index"], row["data_type"]) not in completed_set
```

**효율성**:
- Set lookup: O(1)
- 5M 샘플 필터링도 빠름

### 3. **Tuple Key 사용**

```python
# (task_id, round_index, data_type) 조합으로 unique 식별
("007bbfb7", 0, "original")
("007bbfb7", 0, "hindsight")  # ← 다른 작업으로 취급
("007bbfb7", 1, "original")   # ← 다른 round, 다른 작업
```

**이유**: 같은 task도 round나 data_type이 다르면 별도 처리

---

## 현재 상태 확인

### progress.json 확인

```bash
# 파일 존재 여부
ls -lh /data/hjkim/soar2cot/data/progress.json

# 내용 확인
cat /data/hjkim/soar2cot/data/progress.json | python -m json.tool
```

**현재 상태**: 파일 없음 → 아직 작업 완료된 게 없음

### 첫 실행 후 생성

```bash
# 파이프라인 실행
python -m src.run

# 첫 작업 완료 후 자동 생성됨
cat /data/hjkim/soar2cot/data/progress.json
```

---

## 주의사항

### 1. progress.json 삭제 = 처음부터 다시

```bash
# progress.json 삭제
rm /data/hjkim/soar2cot/data/progress.json

# 재실행 시 모든 작업 다시 수행
python -m src.run  # ← 처음부터
```

### 2. DB는 별도 관리

- `progress.json`: 어떤 작업을 **시도했는지** 추적
- `instructions` 테이블: 실제 **결과** 저장

**중복 저장 방지**: 없음!
- 같은 task를 두 번 처리하면 → DB에 중복 레코드 생성됨
- progress.json이 정확해야 중복 방지됨

### 3. 수동 수정 가능

```bash
# progress.json 직접 편집 가능
vim /data/hjkim/soar2cot/data/progress.json

# 특정 task를 다시 처리하고 싶으면 해당 항목 삭제
```

---

## 고급 기능

### Parquet 업데이트 (현재 미사용)

```python
# src/progress_tracker.py line 93-115
def should_update_parquet(self, interval: int = 5000) -> bool:
    """5000개마다 parquet 업데이트"""

def mark_parquet_updated(self):
    """업데이트 완료 표시"""
```

**용도**: 나중에 SOAR parquet 파일에서 완료된 샘플을 물리적으로 제거하려면 사용

---

## 실제 사용 예시

### 예시 1: 100개 작업 중 50개 완료 후 중단

```bash
# 실행
python -m src.run

# 50개 완료 후 중단 (Ctrl+C)
# progress.json: 50개 기록됨

# 재시작
python -m src.run

# 로그:
Filtered out completed samples remaining=4999950 completed=50
# 51번째 작업부터 재개
```

### 예시 2: 특정 task만 재처리

```bash
# progress.json 편집
vim /data/hjkim/soar2cot/data/progress.json

# "007bbfb7" 항목 삭제
# 저장

# 재실행
python -m src.run

# 007bbfb7만 다시 처리됨 (다른 건 건너뜀)
```

---

## 요약

### ✅ 재시작 시 자동으로

1. `progress.json` 로드
2. 완료된 작업 목록 가져오기
3. SOAR 데이터에서 완료된 것 필터링
4. 남은 작업만 처리

### ✅ 안전성

- 각 작업 완료 즉시 저장
- 중단되어도 최소 손실

### ✅ 효율성

- Set 기반 O(1) lookup
- 5M 샘플도 빠르게 필터링

### ✅ 유연성

- progress.json 수동 편집 가능
- 특정 작업만 재처리 가능

### ⚠️ 주의

- progress.json 삭제 = 처음부터
- DB 중복 방지는 progress.json 정확성에 의존
