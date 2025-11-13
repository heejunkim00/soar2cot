# Structured Instructions 사용법

## 개요

새로운 `StructuredInstructionsResponse`를 사용하면 instruction 생성을 **두 부분으로 명확히 분리**할 수 있습니다:
1. **Input/Output Analysis**: 패턴 분석
2. **Transformation Method**: 변환 단계별 설명

---

## 추가된 구성 요소

### 1. 새로운 Response 모델 (src/main.py)

```python
class StructuredInstructionsResponse(BaseModel):
    """Given the input / output examples, provide analysis and instructions in two separate parts."""

    input_output_analysis: str = Field(
        ...,
        description="Detailed analysis of patterns observed in the input grids and output grids."
    )

    transformation_method: str = Field(
        ...,
        description="Step-by-step instructions for how to transform any input grid into the correct output grid."
    )
```

**기존 모델과 비교**:
```python
# 기존 (InstructionsResponse)
{
    "instructions": "1. Find all colored cells\n2. Create blocks...\n3. ..."
}

# 새로운 (StructuredInstructionsResponse)
{
    "input_output_analysis": "The input grids contain scattered colored cells...",
    "transformation_method": "1. Find all colored cells\n2. Create blocks...\n3. ..."
}
```

### 2. 새로운 프롬프트 (src/prompts.py)

```python
STRUCTURED_PROMPT_WITH_CODE = """
Your task is to analyze the transformation pattern in TWO distinct parts:

## PART 1: INPUT/OUTPUT ANALYSIS
Carefully examine all the training examples and provide a detailed analysis:
- What patterns do you observe in the INPUT grids?
- What patterns do you observe in the OUTPUT grids?
- What relationships exist between inputs and outputs?
...

## PART 2: TRANSFORMATION METHOD
Based on your analysis above, write clear step-by-step instructions:
- The instructions must apply consistently to ALL training examples
- Be general enough to work on new test cases
- Number your steps clearly (1. 2. 3. etc.)
...
"""
```

---

## 사용 방법

### Option 1: 기존 프롬프트 사용 (변경 없음)

```python
from src.main import InstructionsResponse
from src.prompts import INTUITIVE_PROMPT_WITH_CODE

# 기존 방식 그대로
instructions = await get_next_structure(
    structure=InstructionsResponse,
    messages=[{"role": "user", "content": INTUITIVE_PROMPT_WITH_CODE + examples}],
    model=model,
)

# 사용
print(instructions.instructions)
```

### Option 2: 새로운 Structured 프롬프트 사용

```python
from src.main import StructuredInstructionsResponse
from src.prompts import STRUCTURED_PROMPT_WITH_CODE

# 새로운 방식
response = await get_next_structure(
    structure=StructuredInstructionsResponse,
    messages=[{"role": "user", "content": STRUCTURED_PROMPT_WITH_CODE + examples}],
    model=model,
)

# 사용
print("=== ANALYSIS ===")
print(response.input_output_analysis)
print("\n=== TRANSFORMATION ===")
print(response.transformation_method)
```

---

## 실제 적용 예시

### src/run.py의 get_instructions_from_challenge() 수정

기존 코드를 찾아서 수정해야 합니다.

**현재 위치**: `src/run.py` 약 line 600-650

#### 기존 코드 (유지):
```python
async def get_instructions_from_challenge(
    c: Challenge, step: Step, python_code: str
) -> str:
    # ... messages 구성 ...

    instructions = await get_next_structure(
        structure=InstructionsResponse,
        messages=messages,
        model=step.instruction_model,
    )

    return instructions.instructions
```

#### 새로운 옵션 추가:
```python
async def get_structured_instructions_from_challenge(
    c: Challenge, step: Step, python_code: str
) -> tuple[str, str]:  # (analysis, method)
    """Generate structured instructions with separate analysis and method."""
    from src.main import StructuredInstructionsResponse
    from src.prompts import STRUCTURED_PROMPT_WITH_CODE

    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": STRUCTURED_PROMPT_WITH_CODE},
                # ... add examples and python code ...
            ],
        }
    ]

    response, is_truncated = await get_next_structure(
        structure=StructuredInstructionsResponse,
        messages=messages,
        model=step.instruction_model,
    )

    # Log both parts
    _save_llm_interaction(
        function_name="get_structured_instructions_from_challenge",
        task_id=c.task_id,
        messages=messages,
        response=f"ANALYSIS:\n{response.input_output_analysis}\n\nMETHOD:\n{response.transformation_method}",
        metadata=SOAR_METADATA_CONTEXT,
        is_truncated=is_truncated,
    )

    return response.input_output_analysis, response.transformation_method
```

---

## DB 저장 수정

### 기존: instructions 테이블

```sql
CREATE TABLE instructions (
    id TEXT PRIMARY KEY,
    instructions TEXT,  -- 단일 필드
    ...
);
```

### 새로운 옵션: 컬럼 추가

```sql
ALTER TABLE instructions
ADD COLUMN input_output_analysis TEXT,
ADD COLUMN transformation_method TEXT;
```

그러면 두 가지 모드를 지원:
1. 기존 모드: `instructions` 필드 사용
2. Structured 모드: `input_output_analysis` + `transformation_method` 사용

---

## Config 설정으로 전환

새로운 config 옵션 추가:

```python
# src/configs/oss_configs.py

local_gpt_oss_20b_config = RunConfig(
    use_structured_instructions=True,  # ← 새로 추가
    final_follow_model=local_gpt_oss_20b_model,
    steps=[...],
)
```

그리고 `get_instructions_from_challenge()`에서:

```python
if config.use_structured_instructions:
    analysis, method = await get_structured_instructions_from_challenge(...)
    combined_instructions = f"{analysis}\n\n{method}"
else:
    combined_instructions = await get_instructions_from_challenge(...)
```

---

## 빠른 테스트

### 1. Import 확인

```python
# Python 인터프리터에서
from src.main import StructuredInstructionsResponse
from src.prompts import STRUCTURED_PROMPT_WITH_CODE

print(StructuredInstructionsResponse.model_json_schema())
print(STRUCTURED_PROMPT_WITH_CODE[:200])
```

### 2. 간단한 테스트

```python
import asyncio
from src.llms.structured import get_next_structure
from src.llms.models import Model
from src.main import StructuredInstructionsResponse
from src.prompts import STRUCTURED_PROMPT_WITH_CODE

async def test_structured():
    response, is_truncated = await get_next_structure(
        structure=StructuredInstructionsResponse,
        messages=[{
            "role": "user",
            "content": STRUCTURED_PROMPT_WITH_CODE + "\n[example grids here]"
        }],
        model=Model.local_gpt_oss_20b,
    )

    print("=== ANALYSIS ===")
    print(response.input_output_analysis)
    print("\n=== METHOD ===")
    print(response.transformation_method)
    print(f"\nTruncated: {is_truncated}")

asyncio.run(test_structured())
```

---

## 장점

### 1. 명확한 구조
- Analysis와 Method가 명확히 분리
- 각 부분의 역할이 분명함

### 2. 분석 가능
```sql
-- Analysis만 조회
SELECT task_id, input_output_analysis FROM instructions WHERE score = 1.0;

-- Method만 조회
SELECT task_id, transformation_method FROM instructions WHERE score = 1.0;

-- 특정 패턴 분석
SELECT task_id FROM instructions
WHERE input_output_analysis LIKE '%symmetry%';
```

### 3. 품질 향상 가능성
- LLM이 먼저 분석하고 → 그 다음 method 작성
- Two-step thinking으로 더 정확할 수 있음

### 4. 후처리 용이
```python
# Analysis 기반 필터링
good_analysis = df[df['input_output_analysis'].str.len() > 200]

# Method 길이 분석
df['method_steps'] = df['transformation_method'].str.count(r'\d+\.')
```

---

## 기존 코드와의 호환성

### 완전 호환

기존 코드는 **전혀 수정 없이** 그대로 작동합니다:
- `InstructionsResponse` 그대로 존재
- `INTUITIVE_PROMPT_WITH_CODE` 그대로 존재
- 모든 기존 함수 정상 작동

### 선택적 사용

원할 때만 `StructuredInstructionsResponse` 사용:
- 특정 config에서만 활성화
- A/B 테스트 가능
- 점진적 마이그레이션 가능

---

## 다음 단계

### 1. 테스트 실행
```bash
# 간단한 테스트
python -c "from src.main import StructuredInstructionsResponse; print('OK')"
python -c "from src.prompts import STRUCTURED_PROMPT_WITH_CODE; print('OK')"
```

### 2. 실제 적용 (선택)
- Config에 `use_structured_instructions` 플래그 추가
- `get_instructions_from_challenge()` 수정
- DB 스키마 확장 (optional)

### 3. 비교 실험
- 기존 방식 vs Structured 방식
- Training/Test score 비교
- Instruction 품질 비교

---

## 요약

✅ **추가 완료**:
- `StructuredInstructionsResponse` 클래스
- `STRUCTURED_PROMPT_WITH_CODE` 프롬프트

✅ **기존 유지**:
- `InstructionsResponse` (변경 없음)
- `INTUITIVE_PROMPT_WITH_CODE` (변경 없음)

✅ **사용 방법**:
- 기존: 그대로 사용
- 새로운: 원할 때 선택적으로 사용

✅ **구분자**:
- 프롬프트에 "## PART 1", "## PART 2" 명시
- Response 모델에서 두 필드로 분리
