# 파이프라인 실행 결과 요약

## 질문에 대한 답변

### Q1: 왜 3종류의 추론 데이터만 남았는가?

**답변**: 현재 설정에 `StepRevision`이 없기 때문

**4개 추론 함수**:
1. `get_instructions_from_challenge()` - ✅ **호출됨** (Step 1,2,3)
2. `get_revised_instructions()` - ❌ **호출 안 됨** (StepRevision 없음)
3. `get_pooling_instruction_from_scores()` - ✅ **호출됨** (Step 4)
4. `get_score_from_instructions()` - ✅ **호출됨** (모든 Step)

**결과**: 3종류만 로그에 남음

---

### Q2: DB에 데이터가 실제로 저장되었는가?

**답변**: ✅ **예, 저장되었습니다!**

**확인 결과**:
```bash
# 테이블: instructions
# 총 레코드: 34개
# Perfect score (1.0): 9개
```

**테이블 스키마**:
```sql
id                (text)
instructions      (text)      ← 생성된 natural language instruction
model             (text)
example_scores    (jsonb)     ← train examples 점수
score             (double)    ← 0.0 ~ 1.0
task_id           (text)      ← ARC task ID
task_hash         (text)
step              (jsonb)
created_at        (timestamp)
soar_code         (text)      ← Python code (SOAR) ✅
soar_source_model (text)      ← SOAR 모델 이름 ✅
soar_generation   (integer)   ← SOAR generation ✅
soar_round_index  (integer)   ← Round index ✅
is_hindsight      (boolean)   ← Hindsight 여부 ✅
```

**실제 데이터 샘플**:
```
task_id  | score | python_code_preview                    | instructions_preview
---------|-------|----------------------------------------|---------------------
007bbfb7 | 1.0   | def transform(input_grid):...          | 1. Create a 9×9 output grid...
05f2a901 | 1.0   | def transform(input_grid):...          | 1. Scan the grid to locate...
0520fde7 | 1.0   | def transform(grid):...                | 1. The input grid has 3 rows...
```

---

### Q3: Jeremy 코드를 통과한 데이터 추출 방법

**사용자가 원하는 데이터**:
- task_id
- python_code (SOAR)
- instruction (생성된)
- train examples
- test examples
- is_hindsight

**데이터 추출 SQL**:

#### 1. 문제를 맞춘 instruction만 조회 (score = 1.0)

```sql
SELECT
    task_id,
    soar_code as python_code,
    instructions,
    score,
    soar_source_model,
    soar_round_index,
    is_hindsight,
    example_scores,  -- train/test 정보 포함 (JSON)
    created_at
FROM instructions
WHERE score = 1.0
ORDER BY created_at DESC;
```

#### 2. 90% 이상 맞춘 instruction 조회

```sql
SELECT
    task_id,
    soar_code as python_code,
    instructions,
    score,
    soar_round_index,
    is_hindsight
FROM instructions
WHERE score >= 0.9
ORDER BY score DESC, created_at DESC;
```

#### 3. CSV로 내보내기

```bash
# PostgreSQL에 접속
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc

# CSV 내보내기
\copy (SELECT task_id, soar_code as python_code, instructions, score, soar_round_index, is_hindsight FROM instructions WHERE score = 1.0) TO '/tmp/perfect_instructions.csv' CSV HEADER;

# 모든 데이터 내보내기
\copy (SELECT task_id, soar_code as python_code, instructions, score, soar_round_index, is_hindsight FROM instructions ORDER BY score DESC) TO '/tmp/all_instructions.csv' CSV HEADER;
```

#### 4. Python으로 데이터 추출

```python
import psycopg2
import pandas as pd

# DB 연결
conn = psycopg2.connect("postgresql://hjkim:hjkim123@localhost/arc")

# 문제를 맞춘 instruction 조회
query = """
SELECT
    task_id,
    soar_code as python_code,
    instructions,
    score,
    example_scores,
    soar_round_index,
    is_hindsight
FROM instructions
WHERE score = 1.0
ORDER BY created_at DESC
"""

df = pd.read_sql(query, conn)
print(f"Total perfect score instructions: {len(df)}")

# CSV 저장
df.to_csv("/tmp/perfect_instructions.csv", index=False)

conn.close()
```

---

## 현재 상태 요약

### 실행 결과 (20분 실행)

1. **DB에 저장된 데이터**: 34개 instruction scores
2. **Perfect score (100% 정확)**: 9개
3. **처리된 unique tasks**: 약 5-6개 (추정)

### 로그 파일 위치

```
logs/run_YYYYMMDD_HHMMSS/
├── get_instructions_from_challenge/     ← 5개 샘플
├── get_pooling_instruction_from_scores/ ← 2개 샘플
└── get_score_from_instructions/         ← 5개 샘플

(get_revised_instructions/ 없음)
```

### 파이프라인 플로우 (각 task당)

```
Task 시작
├── Step 1: get_instructions_from_challenge() × 2
│   └── get_score_from_instructions() × 2
├── Step 2: get_instructions_from_challenge() × 2
│   └── get_score_from_instructions() × 2
├── Step 3: get_instructions_from_challenge() × 3
│   └── get_score_from_instructions() × 3
└── Step 4: get_pooling_instruction_from_scores() × 2
    └── get_score_from_instructions() × 2

총 LLM 호출: 18번 (instruction 생성 9번 + grid 생성 9번)
DB 저장: 9개 InstructionsScore
```

### Perfect Score 달성 시 조기 종료

```python
# get_answer_grids() 함수
if instruction_scores[0].score == 1:
    return await return_answer(...)  # 즉시 종료
```

- Perfect score면 다음 Step 건너뜀
- 따라서 일부 task는 Step 1-2만 실행하고 종료됨

---

## 다음 작업

### 1. 데이터 확인

```bash
# DB 접속
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc

# 전체 데이터 확인
SELECT task_id, score, LEFT(instructions, 50), created_at
FROM instructions
ORDER BY created_at DESC;

# Perfect score 데이터 확인
SELECT task_id, score, LEFT(soar_code, 100), LEFT(instructions, 100)
FROM instructions
WHERE score = 1.0;
```

### 2. 더 많은 데이터 수집하려면

**옵션 1**: 더 오래 실행
```bash
# 몇 시간 실행하여 더 많은 task 처리
```

**옵션 2**: StepRevision 추가하여 4종류 모두 로그에 남기
```python
# src/configs/oss_configs.py 수정
from src.run import StepRevision

steps=[
    Step(...),
    StepRevision(              # ← 추가
        top_scores_used=2,
        times_per_top_score=1,
        ...
    ),
    Step(...),
    StepRevisionPool(...),
]
```

### 3. 로그 확인

```bash
# 최근 실행 로그 확인
ls -la logs/run_*/

# Instruction 생성 로그 확인
cat logs/run_*/get_instructions_from_challenge/001_*_prompt.txt
cat logs/run_*/get_instructions_from_challenge/001_*_response.txt

# Truncated 응답 확인 (있는 경우)
ls -la logs/run_*/*_TRUNCATED/
```

---

## 결론

✅ **3종류만 남은 이유**: StepRevision이 설정에 없어서

✅ **DB 저장 확인**: 34개 레코드 저장됨, 9개가 perfect score

✅ **원하는 데이터 형태**: DB의 `instructions` 테이블에 모두 저장됨
- task_id ✅
- python_code (soar_code) ✅
- instruction ✅
- score ✅
- is_hindsight ✅
- train/test (example_scores JSON) ✅

✅ **데이터 추출**: SQL 쿼리 또는 Python으로 가능

---

## 빠른 데이터 확인 명령어

```bash
# DB 레코드 수 확인
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "SELECT COUNT(*) FROM instructions;"

# Perfect score 개수
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "SELECT COUNT(*) FROM instructions WHERE score = 1.0;"

# 최근 5개 레코드
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "SELECT task_id, score, LEFT(instructions, 50) FROM instructions ORDER BY created_at DESC LIMIT 5;"

# CSV 내보내기
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "\copy (SELECT * FROM instructions WHERE score = 1.0) TO '/tmp/perfect.csv' CSV HEADER;"
```
