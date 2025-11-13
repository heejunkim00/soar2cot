# Score 계산 방식 상세 설명

## 요약

**Score = Training Examples에서 정답과 일치하는 셀(cell)의 평균 비율**

- **1.0 (100%)**: 모든 training example의 output grid를 완벽하게 생성
- **0.5 (50%)**: 평균적으로 절반의 셀이 정답과 일치
- **0.0 (0%)**: 완전히 틀림

---

## 계산 과정 (3단계)

### 1단계: 개별 Example의 Grid Similarity 계산

**함수**: `get_grid_similarity()` (src/run.py:345-371)

```python
def get_grid_similarity(ground_truth_grid, sample_grid) -> float:
    """
    Calculate similarity as the percentage of cells that match exactly.
    Returns a value between 0.0 (no matches) and 1.0 (perfect match).
    """
    # 1. 그리드 크기 확인
    if len(ground_truth_grid) != len(sample_grid):
        return 0.0  # 크기가 다르면 0점

    # 2. 전체 셀 수 계산
    rows = len(ground_truth_grid)
    cols = len(ground_truth_grid[0])
    total_cells = rows * cols

    # 3. 일치하는 셀 개수 세기
    matching_cells = 0
    for i in range(rows):
        for j in range(cols):
            if ground_truth_grid[i][j] == sample_grid[i][j]:
                matching_cells += 1

    # 4. 일치 비율 반환
    return matching_cells / total_cells
```

**예시**:
```
정답 Grid:              생성된 Grid:
1 2 3                   1 2 3
4 5 6                   4 0 6  ← 5 대신 0 (틀림)
7 8 9                   7 8 9

total_cells = 9
matching_cells = 8 (5만 틀림)
similarity = 8/9 = 0.888... (88.9%)
```

---

### 2단계: 각 Training Example의 Score 계산

**함수**: `get_example_score()` (src/run.py:440-488)

```python
async def get_example_score(
    instructions: str,
    training_examples: list[Example],
    test_example: Example,
    ...
) -> ExampleScore:
    # 1. Instruction에 따라 output grid 생성
    grid_output = await output_grid_from_instructions(
        instructions=instructions,
        training_examples=training_examples,  # 다른 examples 참고
        test_input_grid=test_example.input,   # 이 example의 input
        ...
    )

    # 2. 정답과 비교
    if test_example.output == grid_output:
        similarity_score = 1.0  # 완벽히 일치
    else:
        similarity_score = get_grid_similarity(
            ground_truth_grid=test_example.output,
            sample_grid=grid_output
        )

    # 3. ExampleScore 생성
    return ExampleScore(
        example=test_example,
        response_output_grid=grid_output,
        score=similarity_score,
        ...
    )
```

**과정 설명**:
1. 한 training example을 **test**로 취급
2. 나머지 training examples를 **참고 자료**로 사용
3. Instruction에 따라 LLM이 output grid 생성
4. 정답 output과 비교하여 similarity 계산

---

### 3단계: 전체 Instruction의 Score 계산

**함수**: `score_instructions_on_challenge()` (src/run.py:491-595)

```python
async def score_instructions_on_challenge(
    c: Challenge,
    instructions: str,
    step: Step
) -> InstructionsScore:
    # 1. 각 training example에 대해 score 계산
    futures = []
    for i_train in range(len(c.train)):
        temp_test = c.train[i_train]        # i번째를 test로
        temp_train = c.train[:i_train] + c.train[i_train+1:]  # 나머지를 train으로

        futures.append(
            get_example_score(
                instructions=instructions,
                training_examples=temp_train,
                test_example=temp_test,
                ...
            )
        )

    example_scores: list[ExampleScore] = await asyncio.gather(*futures)

    # 2. 평균 계산
    score = sum(s.score for s in example_scores) / len(example_scores)

    # 3. DB 저장
    instructions_score = InstructionsScore(
        instructions=instructions,
        example_scores=example_scores,
        score=score,  # ← 최종 score
        ...
    )
    await instructions_score.save_to_db(c=c)

    return instructions_score
```

**과정 설명**:
1. **Leave-One-Out Cross Validation** 방식 사용
2. 각 training example을 한 번씩 test로 사용
3. 모든 example_scores의 평균이 최종 score

---

## 구체적인 예시

### Challenge 정보
```python
challenge.train = [
    Example(input=[[1,2]], output=[[3,4]]),  # Example 0
    Example(input=[[5,6]], output=[[7,8]]),  # Example 1
    Example(input=[[9,0]], output=[[1,2]]),  # Example 2
]
```

### Instruction
```
"Add 2 to every cell"
```

### Score 계산 과정

#### Round 1: Example 0을 test로
```
Train: Example 1, 2
Test:  Example 0

LLM에게 제공:
- Instruction: "Add 2 to every cell"
- Train examples: [[5,6]]→[[7,8]], [[9,0]]→[[1,2]]
- Test input: [[1,2]]

LLM 생성: [[3,4]]
정답:      [[3,4]]

비교: 100% 일치
Example 0 score = 1.0
```

#### Round 2: Example 1을 test로
```
Train: Example 0, 2
Test:  Example 1

LLM에게 제공:
- Instruction: "Add 2 to every cell"
- Train examples: [[1,2]]→[[3,4]], [[9,0]]→[[1,2]]
- Test input: [[5,6]]

LLM 생성: [[7,8]]
정답:      [[7,8]]

비교: 100% 일치
Example 1 score = 1.0
```

#### Round 3: Example 2를 test로
```
Train: Example 0, 1
Test:  Example 2

LLM에게 제공:
- Instruction: "Add 2 to every cell"
- Train examples: [[1,2]]→[[3,4]], [[5,6]]→[[7,8]]
- Test input: [[9,0]]

LLM 생성: [[11,2]]  ← 9+2=11, 0+2=2
정답:      [[1,2]]

비교: 1/2 = 50% 일치 (두 번째 셀만 맞음)
Example 2 score = 0.5
```

#### 최종 Score
```python
score = (1.0 + 1.0 + 0.5) / 3 = 0.833... (83.3%)
```

---

## DB 저장 데이터

```sql
SELECT
    task_id,
    instructions,
    score,
    example_scores  -- JSON 배열
FROM instructions
WHERE task_id = 'example_task';
```

**결과**:
```json
{
    "task_id": "example_task",
    "instructions": "Add 2 to every cell",
    "score": 0.8333333,
    "example_scores": [
        {
            "example": {"input": [[1,2]], "output": [[3,4]]},
            "response_output_grid": [[3,4]],
            "score": 1.0
        },
        {
            "example": {"input": [[5,6]], "output": [[7,8]]},
            "response_output_grid": [[7,8]],
            "score": 1.0
        },
        {
            "example": {"input": [[9,0]], "output": [[1,2]]},
            "response_output_grid": [[11,2]],
            "score": 0.5
        }
    ]
}
```

---

## 왜 이 방식을 사용하는가?

### 1. Leave-One-Out Cross Validation
- 각 example을 한 번씩 test로 사용
- **Overfitting 방지**: Instruction이 특정 example만 맞추는 것을 방지
- **일반화 능력 평가**: Instruction이 실제로 규칙을 이해했는지 확인

### 2. Cell-Level Similarity
- **부분 점수**: 완전히 틀린 것보다 일부만 틀린 게 나음
- **세밀한 평가**: 얼마나 가까웠는지 정량화
- **Gradient**: 개선 방향을 알 수 있음

### 3. 평균 Score
- **안정성**: 한 example만 잘 맞춰도 높은 점수를 받지 않음
- **전체 성능**: 모든 examples에서 일관되게 좋아야 높은 점수

---

## Score 해석

| Score | 의미 | 설명 |
|-------|------|------|
| 1.0 | Perfect | 모든 training examples를 100% 정확하게 생성 |
| 0.9 - 0.99 | Excellent | 거의 완벽, 일부 셀만 틀림 |
| 0.7 - 0.89 | Good | 대부분 맞음, 일부 오류 존재 |
| 0.5 - 0.69 | Fair | 절반 정도 맞음 |
| 0.3 - 0.49 | Poor | 대부분 틀림 |
| 0.0 - 0.29 | Very Poor | 거의 완전히 틀림 |

---

## 실제 DB 데이터로 확인

```bash
# Perfect score (1.0) instruction 조회
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "
SELECT
    task_id,
    score,
    LEFT(instructions, 100) as instruction_preview
FROM instructions
WHERE score = 1.0
LIMIT 5;
"

# Score별 분포 확인
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "
SELECT
    CASE
        WHEN score = 1.0 THEN 'Perfect (1.0)'
        WHEN score >= 0.9 THEN 'Excellent (0.9-0.99)'
        WHEN score >= 0.7 THEN 'Good (0.7-0.89)'
        WHEN score >= 0.5 THEN 'Fair (0.5-0.69)'
        ELSE 'Poor (< 0.5)'
    END as score_range,
    COUNT(*) as count
FROM instructions
GROUP BY score_range
ORDER BY MIN(score) DESC;
"
```

---

## 요약

**Score 계산식**:
```
score = Σ(각 training example의 일치하는 셀 비율) / training examples 개수
```

**예시**:
- Training examples: 3개
- Example 1 similarity: 100% (1.0)
- Example 2 similarity: 100% (1.0)
- Example 3 similarity: 50% (0.5)
- **최종 score = (1.0 + 1.0 + 0.5) / 3 = 0.833 (83.3%)**

**Perfect score (1.0)의 의미**:
- 모든 training examples에서 생성된 output grid가 정답과 **완벽히** 일치
- 한 셀이라도 틀리면 1.0이 아님
