# 원하는 데이터 생성 가능 여부 및 현재 상태

## 사용자가 원하는 데이터

```
Training score = 100% (1.0)인 instruction으로
Test input에 대해 guess를 했을 때
실제 정답 grid와 일치한 경우

데이터: (task_id, python_code, instruction, train, test, is_hindsight)
```

---

## 현재 상황

### ✅ 가능합니다!

코드는 이미 구현되어 있습니다:

1. **Training score 계산**: ✅ `instructions` 테이블에 저장됨
2. **Test에 대한 guess 생성**: ✅ 코드에 구현됨
3. **Test 정답과 비교**: ✅ SOAR `predicted_test_output` 사용
4. **Guess 저장**: ⚠️ **테이블이 없어서 실패**

### ❌ 현재 문제

**`guess` 테이블이 존재하지 않아서 데이터가 저장되지 않음**

```bash
$ PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "SELECT * FROM guess;"
ERROR:  relation "guess" does not exist
```

---

## 코드 플로우 (이미 구현됨)

### 1. solve_challenge() 함수 (src/run.py:1031-1110)

```python
async def solve_challenge(
    c: Challenge,
    solution_grids: list[GRID] | None,  # ← SOAR predicted_test_output
    ...
):
    # 1. Guess 생성 (2번 시도)
    first_guess_obj, second_guess_obj = await get_answer_grids(c=c, config=config, python_code=python_code)

    # 2. Test 정답이 있으면 비교
    if solution_grids:
        for guess_obj in [first_guess_obj, second_guess_obj]:
            correct = 0
            total = len(solution_grids)
            guess_scores: list[float] = []

            # 각 test example 비교
            for i in range(len(solution_grids)):
                answer_grid = guess_obj.grids[i]
                solution_grid = solution_grids[i]

                if answer_grid == solution_grid:  # ← 정답 체크!
                    correct += 1
                    guess_scores.append(1.0)
                else:
                    guess_scores.append(0.0)

            # Test score 계산
            score = correct / total

            # DB 저장 시도 (여기서 실패!)
            await guess_obj.save_to_db(scores=guess_scores, avg_score=score)
```

### 2. Guess.save_to_db() 함수 (src/run.py:745-771)

```python
class Guess(BaseModel):
    grids: list[GRID]
    instructions_score: InstructionsScore  # ← Training score 1.0인 instruction
    model: Model

    async def save_to_db(self, avg_score: float, scores: list[float]):
        await conn.execute(
            """
            INSERT INTO guess (id, grids, instructions_score_id, model, avg_score, scores)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            str(uuid.uuid4()),
            json.dumps(self.grids),
            self.instructions_score.id,  # ← Foreign key to instructions table
            self.model.value,
            avg_score,  # ← Test score (0.0 ~ 1.0)
            json.dumps(scores),
        )
```

**저장하려는 데이터**:
- `grids`: Test에 대한 생성된 output grids
- `instructions_score_id`: Training score를 가진 instruction의 ID
- `avg_score`: Test에서 맞춘 비율 (0.0 ~ 1.0)
- `scores`: 각 test example별 점수 `[1.0, 0.0, 1.0, ...]`

### 3. SOAR 데이터에서 test 정답 가져오기

**파일**: `src/run.py` line 1286

```python
await solve_challenge(
    c=challenge,
    solution_grids=task_row["predicted_test_output"],  # ← SOAR 예측값을 정답으로 사용
    config=config,
    python_code=task_row["code"],
)
```

**SOAR 데이터 구조**:
- `code`: Python 코드
- `predicted_test_output`: SOAR 모델이 예측한 test output (정답으로 사용)

---

## 해결 방법

### 1. `guess` 테이블 생성

```sql
CREATE TABLE IF NOT EXISTS guess (
    id TEXT PRIMARY KEY,
    grids JSONB NOT NULL,
    instructions_score_id TEXT NOT NULL REFERENCES instructions(id),
    model TEXT NOT NULL,
    avg_score DOUBLE PRECISION NOT NULL,
    scores JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성 (빠른 조회)
CREATE INDEX idx_guess_instructions_score_id ON guess(instructions_score_id);
CREATE INDEX idx_guess_avg_score ON guess(avg_score);
```

### 2. 테이블 생성 실행

```bash
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc <<EOF
CREATE TABLE IF NOT EXISTS guess (
    id TEXT PRIMARY KEY,
    grids JSONB NOT NULL,
    instructions_score_id TEXT NOT NULL REFERENCES instructions(id),
    model TEXT NOT NULL,
    avg_score DOUBLE PRECISION NOT NULL,
    scores JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_guess_instructions_score_id ON guess(instructions_score_id);
CREATE INDEX IF NOT EXISTS idx_guess_avg_score ON guess(avg_score);
EOF
```

### 3. 파이프라인 재실행

테이블 생성 후 파이프라인을 다시 실행하면:
- Guess.save_to_db()가 정상 작동
- Test 결과가 DB에 저장됨

---

## 원하는 데이터 추출 쿼리

### 조건:
1. Training score = 1.0 (모든 train examples 100% 맞춤)
2. Test score = 1.0 (모든 test examples 100% 맞춤)

### SQL 쿼리:

```sql
SELECT
    i.task_id,
    i.soar_code as python_code,
    i.instructions as instruction,
    i.score as training_score,
    g.avg_score as test_score,
    i.soar_round_index,
    i.is_hindsight,
    g.grids as test_grids,
    g.scores as test_scores_per_example
FROM instructions i
JOIN guess g ON i.id = g.instructions_score_id
WHERE i.score = 1.0          -- Training 100% 정확
  AND g.avg_score = 1.0      -- Test 100% 정확
ORDER BY i.created_at DESC;
```

### Python으로 추출:

```python
import psycopg2
import pandas as pd

conn = psycopg2.connect("postgresql://hjkim:hjkim123@localhost/arc")

query = """
SELECT
    i.task_id,
    i.soar_code as python_code,
    i.instructions,
    i.score as training_score,
    g.avg_score as test_score,
    i.soar_round_index,
    i.is_hindsight
FROM instructions i
JOIN guess g ON i.id = g.instructions_score_id
WHERE i.score = 1.0 AND g.avg_score = 1.0
"""

df = pd.read_sql(query, conn)
df.to_csv("/tmp/perfect_instructions.csv", index=False)

print(f"Total perfect instructions (train+test): {len(df)}")
conn.close()
```

---

## 현재 DB 상태 확인

### Instructions 테이블 (Training score만):

```bash
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "
SELECT COUNT(*) as total, COUNT(*) FILTER (WHERE score = 1.0) as perfect
FROM instructions;
"
```

**결과**:
- Total: 34개
- Perfect (training score = 1.0): 9개

### Guess 테이블 (존재하지 않음):

```bash
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "
SELECT COUNT(*) FROM guess;
"
```

**결과**: `ERROR: relation "guess" does not exist`

---

## 다음 단계

### 옵션 1: 테이블 생성 후 기존 데이터 재처리

1. `guess` 테이블 생성
2. 하지만 **기존에 실행한 데이터는 재처리 불가** (이미 지나감)
3. **새로운 데이터만** guess 테이블에 저장됨

### 옵션 2: 처음부터 재실행 (추천)

1. `guess` 테이블 생성
2. `progress.json` 삭제 (처음부터 다시 시작)
3. 파이프라인 재실행
4. 이번에는 `instructions` + `guess` 모두 저장됨

### 옵션 3: 테이블만 생성하고 계속 실행

1. `guess` 테이블 생성
2. 파이프라인 계속 실행 (중단하지 않고)
3. 이후 처리되는 task들은 guess 저장됨
4. 이전 34개는 guess 없음

---

## 요약

### 질문: 가능한가?

**답**: ✅ **네, 가능합니다!**

코드는 이미 모두 구현되어 있습니다. 단지 `guess` 테이블만 생성하면 됩니다.

### 질문: 현재 그런 데이터가 존재하는가?

**답**: ❌ **아니요, 존재하지 않습니다.**

- `instructions` 테이블: 34개 (9개가 training score = 1.0)
- `guess` 테이블: 존재하지 않음 (테이블 생성 필요)

### 필요한 작업:

1. **`guess` 테이블 생성** (SQL 명령어 위 참조)
2. **파이프라인 재실행** (또는 계속 실행)
3. **데이터 추출** (SQL JOIN으로 training + test 모두 1.0인 경우 필터)

### 생성될 데이터 구조:

```
task_id          | python_code              | instruction                      | training_score | test_score
-----------------|--------------------------|----------------------------------|----------------|------------
007bbfb7         | def transform(...):...   | 1. Create a 9×9 output grid...   | 1.0            | 1.0
05f2a901         | def transform(...):...   | 1. Scan the grid to locate...    | 1.0            | 1.0
```

이 데이터가 바로 사용자님이 원하시는:
- **Training에서 100% 맞추고**
- **Test에서도 100% 맞춘**
- **Instruction**입니다!
