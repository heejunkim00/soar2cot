# Truncated Response Logging 구현 상세

## 목적
토큰 제한으로 인해 잘린 LLM 응답(finish_reason == 'length')을 정확하게 감지하고 별도로 로깅하여 디버깅에 활용

## 주요 변경사항

### 1. GPT-OSS Reasoning Effort 설정 추가

**파일**: `src/llms/structured.py`
**위치**: Line 847

```python
# 추가된 코드
response = await local_vllm_client.chat.completions.create(
    model=model.value,
    messages=messages,
    response_format={"type": "json_object"},
    max_tokens=max_tokens,
    temperature=0.3,
    reasoning_effort="low",  # ← 추가: Reasoning 토큰 감소
    extra_body={"skip_special_tokens": True},
)
```

**효과**:
- GPT-OSS 모델의 사고 과정(reasoning)을 짧게 만들어 토큰 사용량 감소
- 8K→12K→16K→20K 재시도 횟수 감소 예상

---

### 2. _get_next_structure_local_vllm() 반환값 변경

**파일**: `src/llms/structured.py`
**위치**: Line 887

```python
# 수정 전
return output

# 수정 후
return output, (finish_reason == 'length')
```

**변경 이유**: `finish_reason == 'length'`인 경우를 정확하게 감지하기 위해 튜플 반환

---

### 3. get_next_structure() 전체 함수 수정

**파일**: `src/llms/structured.py`
**위치**: Lines 157-260

#### 3.1 함수 시그니처 변경 (Line 161)
```python
# 수정 전
) -> BMType:

# 수정 후
) -> tuple[BMType, bool]:  # Returns (result, is_truncated)
```

#### 3.2 is_truncated 기본값 설정 (Line 179)
```python
async with API_SEMAPHORE:
    is_truncated = False  # Default for non-local models
```

#### 3.3 local_vllm 호출 시 튜플 받기 (Line 208-210)
```python
elif model in [Model.local_gpt_oss_20b]:
    res, is_truncated = await _get_next_structure_local_vllm(
        structure=structure, model=model, messages=messages
    )
```

#### 3.4 반환값 변경 (Line 260)
```python
# 수정 전
return res

# 수정 후
return res, is_truncated
```

#### 3.5 로그에 is_truncated 추가 (Line 257)
```python
log.debug(
    "LLM call completed",
    ...
    is_truncated=is_truncated,  # ← 추가
)
```

---

### 4. ExampleScore 모델에 필드 추가

**파일**: `src/run.py`
**위치**: Line 205

```python
class ExampleScore(BaseModel):
    example: Example
    response_output_grid: GRID
    score: float
    model: Model
    is_truncated: bool = False  # ← 새로 추가
```

**이유**: Grid 생성 응답의 truncation 상태를 추적하기 위해

---

### 5. output_grid_from_instructions() 수정

**파일**: `src/main.py`
**위치**: Lines 325-328

```python
# 수정 전
return (
    await get_next_structure(structure=GridResponse, messages=messages, model=model)
).grid

# 수정 후
grid_response, is_truncated = await get_next_structure(
    structure=GridResponse, messages=messages, model=model
)
return grid_response.grid, is_truncated
```

**변경 이유**: Grid 생성 시에도 truncation 감지 필요

---

### 6. get_example_score() 수정

**파일**: `src/run.py`
**위치**: Lines 451-488

```python
# 수정 전
grid_output = await output_grid_from_instructions(...)

# 수정 후
grid_output, is_truncated = await output_grid_from_instructions(...)

# ExampleScore 생성 시 is_truncated 추가
example_score = ExampleScore(
    example=test_example,
    response_output_grid=grid_output,
    score=similarity_score,
    model=model,
    is_truncated=is_truncated,  # ← 추가
)
```

---

### 7. 4개 Instruction 생성 함수 수정

모든 함수에서 **추측 기반 감지 제거** + **튜플로 받기**

#### 7.1 get_instructions_from_challenge()

**파일**: `src/run.py`
**위치**: Lines 629-642

```python
# 수정 전
instructions = await get_next_structure(
    structure=InstructionsResponse,
    messages=messages,
    model=step.instruction_model,
)
is_truncated = len(instructions.instructions) < 100  # ← 추측 (삭제됨)

# 수정 후
instructions, is_truncated = await get_next_structure(
    structure=InstructionsResponse,
    messages=messages,
    model=step.instruction_model,
)

# Log prompt and response (is_truncated는 그대로 사용)
_save_llm_interaction(
    function_name="get_instructions_from_challenge",
    task_id=c.task_id,
    messages=messages,
    response=instructions.instructions,
    metadata=SOAR_METADATA_CONTEXT,
    is_truncated=is_truncated,
)
```

#### 7.2 get_revised_instructions()

**파일**: `src/run.py`
**위치**: Lines 325-339

```python
# 수정 전
response = await get_next_structure(...)
is_truncated = len(response.revised_instructions) < 100  # ← 추측 (삭제됨)

# 수정 후
response, is_truncated = await get_next_structure(
    structure=ReviseInstructionsResponse,
    messages=messages,
    model=step.instruction_model,
)

_save_llm_interaction(
    function_name="get_revised_instructions",
    task_id=c.task_id,
    messages=messages,
    response=response.revised_instructions,
    metadata={"original_instructions": self.instructions},
    is_truncated=is_truncated,
)
```

#### 7.3 get_pooling_instruction_from_scores()

**파일**: `src/run.py`
**위치**: Lines 718-732

```python
# 수정 전
response = await get_next_structure(...)
is_truncated = len(response.revised_instructions) < 100  # ← 추측 (삭제됨)

# 수정 후
response, is_truncated = await get_next_structure(
    structure=ReviseInstructionsResponse,
    messages=messages,
    model=step.instruction_model,
)

_save_llm_interaction(
    function_name="get_pooling_instruction_from_scores",
    task_id=c.task_id,
    messages=messages,
    response=response.revised_instructions,
    metadata={"num_scores": len(scores)},
    is_truncated=is_truncated,
)
```

#### 7.4 get_score_from_instructions()

**파일**: `src/run.py`
**위치**: Lines 551-566

```python
# 수정 전
if len(example_scores) > 0 and _function_save_counts["get_score_from_instructions"] < MAX_SAMPLES_PER_FUNCTION:
    ...
    _save_llm_interaction(..., metadata={"instructions": instructions})

# 수정 후
if len(example_scores) > 0:  # ← 조건 단순화
    response_grid = example_scores[0].response_output_grid
    response_str = "Output Grid:\n"
    for row in response_grid:
        response_str += " ".join(str(cell) for cell in row) + "\n"
    response_str += f"\nScore: {example_scores[0].score}"

    _save_llm_interaction(
        function_name="get_score_from_instructions",
        task_id=c.task_id,
        messages=sample_messages,
        response=response_str,
        metadata={"instructions": instructions},
        is_truncated=example_scores[0].is_truncated,  # ← ExampleScore에서 가져옴
    )
```

**참고**: 이 함수는 `output_grid_from_instructions()` → `get_example_score()` 를 통해 `is_truncated`를 받아옴

---

### 8. return_answer() 함수 수정

**파일**: `src/run.py`
**위치**: Lines 820-835

```python
# 수정 전
_final_output_grids: list[GRID] = await asyncio.gather(*futures, return_exceptions=True)

for i, g in enumerate(_final_output_grids):
    if isinstance(g, Exception):
        ...
    else:
        final_output_grids_and_scores.append((g, scores_to_use[i]))

# 수정 후
_final_output_grids_with_truncation: list[tuple[GRID, bool]] = await asyncio.gather(
    *futures, return_exceptions=True
)

for i, result in enumerate(_final_output_grids_with_truncation):
    if isinstance(result, Exception):
        log.error(...)
    else:
        grid, is_truncated = result  # ← 튜플 언팩
        final_output_grids_and_scores.append((grid, scores_to_use[i]))
```

**변경 이유**: `output_grid_from_instructions()`가 튜플을 반환하므로 언팩 필요

---

## 로깅 디렉토리 구조

```
logs/run_YYYYMMDD_HHMMSS/
├── get_instructions_from_challenge/          # 정상 응답 (처음 5개)
│   ├── 001_task_{task_id}_prompt.txt
│   └── 001_task_{task_id}_response.txt
├── get_instructions_from_challenge_TRUNCATED/ # 토큰 제한 응답 (처음 10개)
│   ├── TRUNC_001_task_{task_id}_prompt.txt
│   └── TRUNC_001_task_{task_id}_response.txt
├── get_revised_instructions/
├── get_revised_instructions_TRUNCATED/
├── get_pooling_instruction_from_scores/
├── get_pooling_instruction_from_scores_TRUNCATED/
├── get_score_from_instructions/
└── get_score_from_instructions_TRUNCATED/
```

## 제거된 코드

### 추측 기반 truncation 감지 (4개 함수)
```python
# 모두 제거됨 ❌
is_truncated = len(instructions.instructions) < 100
is_truncated = len(response.revised_instructions) < 100
is_truncated = len(response_grid) == 0 or len(response_str) < 50
```

### 유지된 코드

`_truncated_save_counts` 딕셔너리는 **유지**됨 (Line 70-75):
```python
_truncated_save_counts = {
    "get_instructions_from_challenge": 0,
    "get_revised_instructions": 0,
    "get_pooling_instruction_from_scores": 0,
    "get_score_from_instructions": 0,
}
MAX_TRUNCATED_SAMPLES = 10
```

**이유**: `_save_llm_interaction()` 함수에서 truncated 응답 개수를 카운트하는 데 사용

---

## 전체 데이터 플로우

```
1. LLM 호출 (4가지 종류)
   ↓
2. _get_next_structure_local_vllm()
   → finish_reason 체크 → (output, finish_reason == 'length')
   ↓
3. get_next_structure()
   → local_vllm이면 튜플 받기
   → 다른 모델이면 is_truncated=False
   → (output, is_truncated) 반환
   ↓
4. 각 함수에서 튜플 언팩
   → instructions, is_truncated = await get_next_structure(...)
   ↓
5. _save_llm_interaction() 호출
   → is_truncated=True면 _TRUNCATED/ 디렉토리에 저장
   → is_truncated=False면 일반 디렉토리에 저장
```

---

## 테스트 방법

1. vLLM 서버 시작:
   ```bash
   ./start_vllm_server.sh
   ```

2. 파이프라인 실행:
   ```bash
   python -m src.run
   ```

3. 로그 확인:
   ```bash
   ls -la logs/run_*/
   ls -la logs/run_*/*_TRUNCATED/
   ```

4. Truncated 응답 내용 확인:
   ```bash
   cat logs/run_*/get_instructions_from_challenge_TRUNCATED/TRUNC_001_*_response.txt
   ```

---

## 주의사항

1. **다른 모델 (OpenAI, Anthropic 등)은 항상 is_truncated=False**
   - finish_reason 정보를 추출하지 않음
   - 필요시 각 `_get_next_structure_*()` 함수도 튜플 반환하도록 수정 필요

2. **is_truncated는 정확한 정보**
   - GPT-OSS에서: finish_reason == 'length'인 경우만 True
   - 응답 길이나 내용으로 추측하지 않음

3. **기존 로깅은 그대로 유지**
   - 정상 응답: 처음 5개 저장
   - Truncated 응답: 처음 10개 별도 저장
   - 두 개는 독립적으로 카운트됨

---

## 수정된 파일 목록

1. `src/llms/structured.py`
   - `_get_next_structure_local_vllm()`: 튜플 반환
   - `get_next_structure()`: 튜플 반환, is_truncated 처리
   - `reasoning_effort="low"` 추가

2. `src/main.py`
   - `output_grid_from_instructions()`: 튜플 반환

3. `src/run.py`
   - `ExampleScore`: `is_truncated` 필드 추가
   - `get_example_score()`: 튜플 받기
   - `get_instructions_from_challenge()`: 추측 제거, 튜플 받기
   - `get_revised_instructions()`: 추측 제거, 튜플 받기
   - `get_pooling_instruction_from_scores()`: 추측 제거, 튜플 받기
   - `get_score_from_instructions()`: ExampleScore에서 is_truncated 사용
   - `return_answer()`: 튜플 언팩 처리

---

## 예상 결과

1. **토큰 사용량 감소**: `reasoning_effort="low"` 덕분에 더 적은 토큰 사용
2. **재시도 감소**: 8K로 충분한 경우가 많아져서 12K, 16K 재시도 감소
3. **정확한 디버깅**: finish_reason 기반으로 정확히 truncated 응답만 저장
4. **디버깅 용이성**: `_TRUNCATED/` 디렉토리에서 문제 응답만 모아서 확인 가능
