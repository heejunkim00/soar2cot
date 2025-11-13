# Numpy Array Bug 수정

## 문제
모든 task가 실패하여 progress.json이 생성되지 않음

## 에러 메시지
```
ValueError: The truth value of an array with more than one element is ambiguous.
Use a.any() or a.all()
```

## 근본 원인

### 위치: `/data/hjkim/soar2cot/src/run.py` 라인 1091

```python
# 수정 전 (에러 발생)
if solution_grids:
    final_scores: list[float] = []
```

**문제점**:
- `solution_grids`가 numpy array일 경우
- `if solution_grids:`는 boolean 변환 불가능
- numpy array는 `a.any()` 또는 `a.all()` 사용해야 함

### 추가 문제: 라인 1296, 1310

```python
# 라인 1296
solution_grids = [task_row["predicted_test_output"]]  # 변수 생성

# 라인 1310
solution_grids=task_row["predicted_test_output"],  # 변수 사용 안함!
```

**문제점**:
- 변수를 만들었지만 사용하지 않음
- 불필요한 리스트 감싸기

## 수정 내용

### 수정 1: Boolean 체크 수정

```python
# 수정 전
if solution_grids:

# 수정 후
if solution_grids is not None and len(solution_grids) > 0:
```

### 수정 2: 변수 일관성 수정

```python
# 수정 전
solution_grids = [task_row["predicted_test_output"]]  # 리스트로 감쌈
...
solution_grids=task_row["predicted_test_output"],  # 변수 안 씀

# 수정 후
solution_grids = task_row["predicted_test_output"]  # 리스트 제거
...
solution_grids=solution_grids,  # 변수 사용
```

## 영향

### 수정 전
- 모든 task가 ValueError로 실패
- progress.json 생성 안됨
- 로그에만 에러 기록

### 수정 후
- task 정상 완료
- progress.json 생성됨
- 통계 추적 가능

## 테스트 방법

```bash
# 기존 프로세스 종료
pkill -f "python -m src.run"

# 새로 시작
python -m src.run

# progress.json 확인
watch -n 5 cat /data/hjkim/soar2cot/data/progress.json
```

## 되돌리기

```bash
cd /data/hjkim/soar2cot
git diff src/run.py
git checkout -- src/run.py  # 되돌리기
```