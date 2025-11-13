# 파이프라인 플로우 분석 및 3종류만 로그에 남은 이유

## 질문
- 4개의 추론 함수가 있는데 왜 3종류만 로그에 남았는가?
- DB에 실제로 데이터가 저장되었는가?
- 최종적으로 문제를 맞춘 instruction은 어떻게 저장되는가?

---

## 4개의 LLM 추론 함수

1. `get_instructions_from_challenge()` - 초기 instruction 생성 (SOAR code 포함)
2. `get_revised_instructions()` - 기존 instruction 개선
3. `get_pooling_instruction_from_scores()` - 여러 instruction 합성
4. `get_score_from_instructions()` - instruction에 따라 grid 생성 및 점수 계산

---

## 현재 GPT-OSS 설정 (src/configs/oss_configs.py)

```python
local_gpt_oss_20b_config = RunConfig(
    final_follow_model=local_gpt_oss_20b_model,
    final_follow_times=3,
    max_concurrent_tasks=4,
    steps=[
        Step(                    # Step 1
            instruction_model=local_gpt_oss_20b_model,
            follow_model=local_gpt_oss_20b_model,
            times=2,             # instruction 2개 생성
            ...
        ),
        Step(                    # Step 2
            times=2,             # instruction 2개 생성
            ...
        ),
        Step(                    # Step 3
            times=3,             # instruction 3개 생성
            ...
        ),
        StepRevisionPool(        # Step 4
            top_scores_used=3,   # 상위 3개 instruction 사용
            times=2,             # pooling된 instruction 2개 생성
            ...
        ),
    ],
)
```

**중요**: `StepRevision`이 없음! `Step`과 `StepRevisionPool`만 있음.

---

## 파이프라인 실행 플로우

### 메인 함수: `get_answer_grids()`

```python
async def get_answer_grids(c: Challenge, config: RunConfig, python_code: str):
    instruction_scores: list[InstructionsScore] = []

    for step in config.steps:
        if isinstance(step, Step):
            # ===== Step 1, 2, 3에서 실행 =====
            instruction_scores.extend(
                await get_instruction_scores(c=c, step=step, python_code=python_code)
            )

        elif isinstance(step, StepRevisionPool):
            # ===== Step 4에서 실행 =====
            futures = []
            for _ in range(step.times):
                futures.append(
                    get_pooling_instruction_from_scores(
                        c=c,
                        scores=instruction_scores[0:step.top_scores_used],
                        step=step,
                    )
                )
            revised_instructions = await asyncio.gather(*futures)

            # 생성된 instruction들을 평가
            for revised_instruction in revised_instructions:
                futures.append(
                    get_score_from_instructions(
                        c=c,
                        instructions=revised_instruction,
                        step=step
                    )
                )
            new_instruction_scores = await asyncio.gather(*futures)
            instruction_scores = [*new_instruction_scores, *instruction_scores]

        elif isinstance(step, StepRevision):
            # ===== 현재 설정에는 없음! =====
            # get_revised_instructions() 여기서 호출됨
            pass

        # 점수순 정렬
        instruction_scores = sorted(instruction_scores, key=lambda x: x.score, reverse=True)

        # Perfect score면 조기 종료
        if instruction_scores[0].score == 1:
            return await return_answer(...)

    # 최종 답변 생성
    return await return_answer(c=c, scores=instruction_scores, config=config, step=prev_step)
```

### Step 1, 2, 3 실행: `get_instruction_scores()`

```python
async def get_instruction_scores(c: Challenge, step: Step, python_code: str):
    futures = [
        get_instruction_score_from_challenge(c=c, step=step, python_code=python_code)
        for _ in range(step.times)  # Step 1,2: 2번, Step 3: 3번
    ]
    return await asyncio.gather(*futures)
```

### `get_instruction_score_from_challenge()`

```python
async def get_instruction_score_from_challenge(c: Challenge, step: Step, python_code: str):
    # 1. Instruction 생성 (함수 1)
    instructions = await get_instructions_from_challenge(
        c=c,
        step=step,
        python_code=python_code
    )

    # 2. Instruction 평가 (함수 4)
    return await get_score_from_instructions(
        c=c,
        instructions=instructions,
        step=step
    )
```

---

## 왜 3종류만 로그에 남았는가?

### 호출되는 함수

| Step | 함수 1 (get_instructions_from_challenge) | 함수 2 (get_revised_instructions) | 함수 3 (get_pooling_instruction_from_scores) | 함수 4 (get_score_from_instructions) |
|------|------------------------------------------|-----------------------------------|---------------------------------------------|--------------------------------------|
| Step 1 (times=2) | ✅ 2번 | ❌ | ❌ | ✅ 2번 |
| Step 2 (times=2) | ✅ 2번 | ❌ | ❌ | ✅ 2번 |
| Step 3 (times=3) | ✅ 3번 | ❌ | ❌ | ✅ 3번 |
| Step 4 (StepRevisionPool, times=2) | ❌ | ❌ | ✅ 2번 | ✅ 2번 |

**총 호출 횟수**:
- 함수 1 (get_instructions_from_challenge): 2+2+3 = **7번**
- 함수 2 (get_revised_instructions): **0번** ← StepRevision이 없어서!
- 함수 3 (get_pooling_instruction_from_scores): **2번**
- 함수 4 (get_score_from_instructions): 2+2+3+2 = **9번**

### 로깅 제한

`src/run.py` (lines 67-68):
```python
MAX_SAMPLES_PER_FUNCTION = 5  # 각 함수당 처음 5개만 저장
MAX_TRUNCATED_SAMPLES = 10     # Truncated 응답은 처음 10개 저장
```

**20분 실행 후 예상 로그**:
- `get_instructions_from_challenge/`: 최대 5개 (7번 호출 중 처음 5개)
- `get_revised_instructions/`: **0개** (호출 안 됨)
- `get_pooling_instruction_from_scores/`: 2개 (2번만 호출됨)
- `get_score_from_instructions/`: 최대 5개 (9번 호출 중 처음 5개)

**결과**: **3종류**만 로그에 남음!

---

## DB 저장 로직

### 1. InstructionsScore 저장

**파일**: `src/run.py` - `get_score_from_instructions()` 함수

```python
async def get_score_from_instructions(
    c: Challenge,
    instructions: str,
    step: Step
) -> InstructionsScore:
    # ... instruction 평가 ...

    instructions_score = InstructionsScore(
        id=str(uuid.uuid4()),
        instructions=instructions,
        example_scores=example_scores,
        score=score,
        model=step.instruction_model,
        step=step,
        # SOAR metadata
        soar_code=SOAR_METADATA_CONTEXT.get("soar_code"),          # ✅ Python code
        soar_source_model=SOAR_METADATA_CONTEXT.get("soar_source_model"),
        soar_generation=SOAR_METADATA_CONTEXT.get("soar_generation"),
        soar_round_index=SOAR_METADATA_CONTEXT.get("soar_round_index"),
        is_hindsight=SOAR_METADATA_CONTEXT.get("is_hindsight", False),
    )

    # DB 저장
    await instructions_score.save_to_db(c=c)

    return instructions_score
```

**저장되는 데이터** (`InstructionsScore` 테이블):
- `instructions`: 생성된 natural language instruction
- `score`: 점수 (0.0 ~ 1.0)
- `soar_code`: Python code (SOAR)
- `soar_source_model`: SOAR 모델 이름
- `soar_generation`: SOAR generation
- `soar_round_index`: Round index
- `is_hindsight`: False (original data)
- `task_id`: ARC task ID
- `train`: Train examples
- `test`: Test inputs

### 2. Guess 저장 (최종 답변)

**파일**: `src/run.py` - `solve_challenge()` 함수

```python
async def solve_challenge(c: Challenge, python_code: str, ...):
    # 최종 답변 생성
    first_guess_obj, second_guess_obj = await get_answer_grids(
        c=c,
        config=config,
        python_code=python_code
    )

    # DB 저장
    for guess_obj in [first_guess_obj, second_guess_obj]:
        await guess_obj.save_to_db(scores=guess_scores, avg_score=score)
```

`Guess` 객체는:
- `grids`: 최종 생성된 output grids
- `instructions_score`: 사용된 InstructionsScore (instruction 포함)
- `model`: 모델 이름
- `avg_score`: 평균 점수

---

## DB 확인 방법

### PostgreSQL 접속

```bash
# .env 파일에서 NEON_DSN 확인
cat .env | grep NEON_DSN

# psql로 접속
psql "$NEON_DSN"
```

### 저장된 데이터 확인

```sql
-- InstructionsScore 테이블 확인
SELECT
    task_id,
    instructions,
    score,
    soar_code,
    soar_source_model,
    soar_round_index,
    is_hindsight,
    created_at
FROM instructions_scores
ORDER BY created_at DESC
LIMIT 10;

-- 문제를 맞춘 instruction만 조회 (score = 1.0)
SELECT
    task_id,
    instructions,
    score,
    soar_code
FROM instructions_scores
WHERE score = 1.0
ORDER BY created_at DESC;

-- 테이블별 레코드 수 확인
SELECT
    'instructions_scores' as table_name,
    COUNT(*) as count
FROM instructions_scores
UNION ALL
SELECT
    'guesses' as table_name,
    COUNT(*) as count
FROM guesses;
```

### Progress Tracker 확인

```bash
# progress.json 확인
cat /data/hjkim/soar2cot/progress.json

# 완료된 task 수 확인
python -c "
import json
with open('progress.json') as f:
    data = json.load(f)
    print(f'Completed tasks: {len(data.get(\"completed\", []))}')
    print(f'First 5 completed:', data.get('completed', [])[:5])
"
```

---

## 최종 데이터 추출 쿼리

### 목표 데이터: 문제를 맞춘 instruction

```sql
-- 사용자가 원하는 형태의 데이터
SELECT
    task_id,
    soar_code as python_code,
    instructions,
    score,
    soar_source_model,
    soar_round_index,
    is_hindsight,
    created_at
FROM instructions_scores
WHERE score >= 0.9  -- 90% 이상 맞춘 경우
ORDER BY score DESC, created_at DESC;

-- CSV로 내보내기
\copy (SELECT task_id, soar_code, instructions, score, soar_round_index, is_hindsight FROM instructions_scores WHERE score >= 0.9) TO '/tmp/correct_instructions.csv' CSV HEADER;
```

---

## 왜 get_revised_instructions()가 호출 안 되는가?

**이유**: 현재 설정에 `StepRevision`이 없기 때문

### StepRevision 추가하려면?

`src/configs/oss_configs.py` 수정:

```python
from src.run import StepRevision  # 추가

local_gpt_oss_20b_config = RunConfig(
    steps=[
        Step(...),
        Step(...),
        StepRevision(              # ← 추가
            top_scores_used=2,
            times_per_top_score=1,
            instruction_model=local_gpt_oss_20b_model,
            follow_model=local_gpt_oss_20b_model,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(...),
        StepRevisionPool(...),
    ],
)
```

---

## 요약

### 3종류만 남은 이유
- `StepRevision`이 설정에 없어서 `get_revised_instructions()`가 호출 안 됨
- 따라서 4개 중 3개 함수만 실행됨

### DB 저장 여부
- ✅ **모든 InstructionsScore는 DB에 저장됨**
- 저장 위치: `instructions_scores` 테이블
- 포함 데이터: task_id, python_code, instruction, score, train, test, is_hindsight 등

### 문제를 맞춘 instruction 확인
```sql
SELECT * FROM instructions_scores WHERE score = 1.0;
```

### 다음 단계
1. DB 접속해서 데이터 확인
2. `progress.json` 확인하여 완료된 task 수 확인
3. 원하는 데이터를 CSV로 export
