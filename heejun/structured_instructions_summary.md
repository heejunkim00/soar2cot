# Structured Instructions 추가 완료

## 변경 사항 요약

### ✅ 추가된 파일/코드

#### 1. `src/main.py` - 새로운 Response 모델
```python
class StructuredInstructionsResponse(BaseModel):
    input_output_analysis: str  # Part 1: 패턴 분석
    transformation_method: str   # Part 2: 변환 단계
```

#### 2. `src/prompts.py` - 새로운 프롬프트
```python
STRUCTURED_PROMPT_WITH_CODE = """
## PART 1: INPUT/OUTPUT ANALYSIS
- 패턴 관찰 및 분석 요청

## PART 2: TRANSFORMATION METHOD
- 단계별 변환 방법 요청
"""
```

### ✅ 기존 유지 (변경 없음)

- `InstructionsResponse` - 그대로 사용 가능
- `INTUITIVE_PROMPT_WITH_CODE` - 그대로 사용 가능
- 모든 기존 코드 - 정상 작동

---

## 출력 형식

### 기존 방식
```json
{
  "instructions": "1. Find all colored cells\n2. Create blocks for each cell\n3. Arrange blocks in a 3x3 pattern"
}
```

### 새로운 방식 (Structured)
```json
{
  "input_output_analysis": "The input grids contain scattered colored cells in a 7x7 grid. The output grids show these cells expanded into 3x3 blocks arranged in a 9x9 grid. The spatial relationships are preserved - if a cell is at position (r,c) in input, its corresponding block appears at (3r, 3c) in output.",

  "transformation_method": "1. Identify all non-zero colored cells in the input grid and note their positions and colors.\n2. For each colored cell at position (row, col) with value V:\n   a. Calculate the top-left position in output: (3*row, 3*col)\n   b. Create a 3x3 block filled with value V\n   c. Place this block at the calculated position\n3. Fill all remaining cells with 0 (background)."
}
```

---

## 사용 방법

### 기존 코드 (변경 없음)
```python
from src.main import InstructionsResponse
from src.prompts import INTUITIVE_PROMPT_WITH_CODE

instructions, is_truncated = await get_next_structure(
    structure=InstructionsResponse,
    messages=messages,
    model=model,
)

print(instructions.instructions)
```

### 새로운 Structured 방식
```python
from src.main import StructuredInstructionsResponse
from src.prompts import STRUCTURED_PROMPT_WITH_CODE

response, is_truncated = await get_next_structure(
    structure=StructuredInstructionsResponse,
    messages=messages,
    model=model,
)

print("ANALYSIS:", response.input_output_analysis)
print("METHOD:", response.transformation_method)
```

---

## 구분자 명시

프롬프트에서 명확하게 구분:

```
## PART 1: INPUT/OUTPUT ANALYSIS
Carefully examine all the training examples and provide a detailed analysis:
- What patterns do you observe in the INPUT grids?
- What patterns do you observe in the OUTPUT grids?
...

## PART 2: TRANSFORMATION METHOD
Based on your analysis above, write clear step-by-step instructions:
1. [First step]
2. [Second step]
...
```

LLM이 두 section을 명확히 구분하여 작성하도록 지시됨.

---

## 장점

### 1. 명확한 구조
- 분석 단계와 실행 단계가 분리
- 각 부분의 목적이 명확

### 2. 품질 향상 가능성
- 먼저 분석 → 그 다음 방법 제시
- Two-step thinking으로 더 정확할 수 있음

### 3. 분석 용이
```sql
-- 분석만 조회
SELECT task_id, input_output_analysis FROM instructions;

-- 특정 패턴 검색
SELECT task_id FROM instructions
WHERE input_output_analysis LIKE '%symmetry%';
```

### 4. 유연성
- 기존 방식과 새 방식 둘 다 사용 가능
- Config로 전환 가능
- A/B 테스트 가능

---

## 테스트 확인

```bash
# Import 테스트
python -c "from src.main import StructuredInstructionsResponse; print('OK')"
python -c "from src.prompts import STRUCTURED_PROMPT_WITH_CODE; print('OK')"

# 둘 다 OK 출력됨 ✅
```

---

## 다음 단계 (선택 사항)

### Option 1: 그대로 사용
현재 코드를 그대로 두고 원할 때만 Structured 방식 사용

### Option 2: Config 추가
```python
# src/configs/oss_configs.py
RunConfig(
    use_structured_instructions=True,  # 새 방식 활성화
    ...
)
```

### Option 3: 함수 추가
```python
# src/run.py에 새 함수 추가
async def get_structured_instructions_from_challenge(...):
    response = await get_next_structure(
        structure=StructuredInstructionsResponse,
        ...
    )
    return response
```

### Option 4: DB 스키마 확장
```sql
ALTER TABLE instructions
ADD COLUMN input_output_analysis TEXT,
ADD COLUMN transformation_method TEXT;
```

---

## 파일 위치

1. **코드 변경**:
   - `/data/hjkim/soar2cot/src/main.py` - `StructuredInstructionsResponse` 추가
   - `/data/hjkim/soar2cot/src/prompts.py` - `STRUCTURED_PROMPT_WITH_CODE` 추가

2. **문서**:
   - `/data/hjkim/soar2cot/heejun/structured_instructions_usage.md` - 사용법 상세
   - `/data/hjkim/soar2cot/heejun/structured_instructions_summary.md` - 요약 (현재 파일)

---

## 요약

✅ **완료됨**:
- 새로운 `StructuredInstructionsResponse` 모델 추가
- 새로운 `STRUCTURED_PROMPT_WITH_CODE` 프롬프트 추가
- 명확한 구분자 ("## PART 1", "## PART 2")
- 기존 코드 100% 호환성 유지

✅ **사용 가능**:
- Import 테스트 통과
- 언제든 사용 가능한 상태

✅ **기존 유지**:
- 모든 기존 프롬프트/모델 그대로
- 기존 파이프라인 정상 작동
