# ARC vs SOAR 데이터셋 비교 분석

**작성일**: 2025-11-11
**데이터 위치**:
- 기존 ARC: `/home/ubuntu/arc-lang-public/data/arc-prize-2024/`
- SOAR: `/data/m-soar/soar_arc_train_5M/`

---

## 1. 개요

### 기존 ARC 데이터셋
- **목적**: ARC (Abstraction and Reasoning Corpus) 문제 정의
- **형식**: JSON
- **내용**: 각 task의 input-output 예제

### SOAR 데이터셋
- **목적**: LLM의 ARC 문제 해결 학습 데이터
- **형식**: Parquet (compressed columnar format)
- **내용**: 모델이 생성한 Python 코드와 예측 결과

---

## 2. 데이터 구조 비교

### 2.1 기존 ARC 데이터셋 구조

**파일 형식**: JSON

```json
{
  "007bbfb7": {
    "train": [
      {
        "input": [[0, 7, 7], [7, 7, 7], [0, 7, 7]],
        "output": [[0, 0, 0, 0, 7, 7, 0, 7, 7], ...]
      },
      {
        "input": [[4, 0, 4], [0, 0, 0], [0, 4, 0]],
        "output": [[4, 0, 4, 0, 0, 0, 4, 0, 4], ...]
      }
      // ... 총 5개 train examples
    ],
    "test": [
      {
        "input": [[7, 0, 7], [7, 0, 7], [7, 7, 0]]
        // output은 solutions 파일에 별도 저장
      }
    ]
  }
  // ... 400개 tasks
}
```

**특징**:
- 각 task는 고유 ID (예: `007bbfb7`)
- Train examples: input-output 쌍 (보통 3-5개)
- Test examples: input만 제공 (정답은 solutions 파일)
- 순수 grid 데이터만 포함

---

### 2.2 SOAR 데이터셋 구조

**파일 형식**: Parquet (5개 파일로 분할)

**각 row의 필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `task_id` | string | ARC task ID (예: "007bbfb7") |
| `model` | string | 사용된 모델 (예: "Mistral-Large-Instruct-2407") |
| `generation` | int | 생성 반복 횟수 |
| `code` | string | Python 변환 함수 |
| `correct_train_input` | list[bool] | 각 train example 정답 여부 ⭐ |
| `predicted_train_output` | list[list[list[int]]] | 모델이 예측한 train outputs |
| `correct_test_input` | list[bool] | 각 test example 정답 여부 ⭐ |
| `predicted_test_output` | list[list[list[int]]] | 모델이 예측한 test outputs |

⭐ **중요**: `correct_train_input`과 `correct_test_input` 필드로 원본 ARC vs Hindsight Relabeling 데이터를 구분할 수 있습니다 (자세한 내용은 섹션 2.3 참조).

**예시 데이터**:

```python
{
  "task_id": "007bbfb7",
  "model": "Mistral-Large-Instruct-2407",
  "generation": 0,
  "code": """
def transform(input_grid: list[list[int]]) -> list[list[int]]:
    output_size = 9
    output_grid = [[0] * output_size for _ in range(output_size)]

    def place_cells(x, y, val):
        for i in range(3):
            for j in range(3):
                output_grid[y * 3 + i][x * 3 + j] = input_grid[i][j] if input_grid[i][j] == val else 0

    for i in range(3):
        for j in range(3):
            if input_grid[i][j] != 0:
                place_cells(j, i, input_grid[i][j])
    return output_grid
  """,
  "correct_train_input": [True, True, True, True, True],
  "predicted_train_output": [grid1, grid2, grid3, grid4, grid5],
  "correct_test_input": [True],
  "predicted_test_output": [grid1]
}
```

**특징**:
- 하나의 task에 대해 **여러 solution attempts** 저장
- 모델이 **생성한 Python 코드** 포함
- 각 solution의 **정답 여부** 명시
- **성공/실패 케이스** 모두 포함

---

## 3. 통계 비교

| 항목 | 기존 ARC 데이터셋 | SOAR 데이터셋 |
|------|------------------|---------------|
| **총 파일 크기** | ~50MB | 1.8GB (compressed) |
| **총 레코드 수** | 400 tasks | 4,926,487 rows |
| **Task당 레코드** | 1개 | 평균 ~12,316개 |
| **예시 (task 007bbfb7)** | 1개 | 2,907개 solutions |
| **데이터 포맷** | JSON | Parquet |
| **압축 여부** | 없음 | Snappy 압축 |

---

## 4. 포함 내용 비교

| 내용 | 기존 ARC | SOAR |
|------|----------|------|
| Input grids | ✓ | ✗ (code로 재현) |
| Output grids (정답) | ✓ | ✗ |
| Output grids (예측) | ✗ | ✓ |
| Python code | ✗ | ✓ |
| 모델 정보 | ✗ | ✓ |
| 정답 여부 | N/A | ✓ |
| Generation 메타데이터 | ✗ | ✓ |

---

## 5. 사용 목적 차이

### 기존 ARC 데이터셋
**용도**: 문제 정의 및 평가
- ARC 문제의 input → output 관계 제시
- 모델 평가용 ground truth
- 연구자가 문제를 이해하는 참고 자료

**작업 흐름**:
```
문제 (ARC JSON) → 모델 추론 → 예측 결과 → 정답과 비교
```

---

### SOAR 데이터셋
**용도**: LLM 학습 데이터
- Code generation 훈련
- Evolutionary algorithm의 학습
- 성공/실패 패턴 분석
- Self-improving 시스템 구축

**작업 흐름**:
```
문제 (ARC) → 모델이 코드 생성 → 코드 실행 → 결과 평가 → 학습 데이터로 저장 (SOAR)
```

**SOAR 프레임워크**:
1. **Evolutionary Search**: LLM이 수천 개 후보 코드 생성
2. **Hindsight Relabeling**: 실패한 코드도 "다른 문제의 정답"으로 활용
3. **Fine-tuning**: 생성된 데이터로 모델 재학습
4. **반복**: 개선된 모델로 다시 1단계부터 수행

---

## 6. 실제 데이터 예시

### 6.1 기존 ARC 예시

**Task**: 007bbfb7

**Train Example 1**:
- Input: `[[0,7,7], [7,7,7], [0,7,7]]` (3×3 grid)
- Output: 9×9 grid (특정 패턴으로 확장)

**Test**:
- Input: `[[7,0,7], [7,0,7], [7,7,0]]`
- Output: ? (모델이 예측해야 함)

---

### 6.2 SOAR 예시

**같은 Task (007bbfb7)에 대한 solution**:

```python
def transform(input_grid: list[list[int]]) -> list[list[int]]:
    output_size = 9
    output_grid = [[0] * output_size for _ in range(output_size)]

    def place_cells(x, y, val):
        for i in range(3):
            for j in range(3):
                output_grid[y * 3 + i][x * 3 + j] = input_grid[i][j] if input_grid[i][j] == val else 0

    for i in range(3):
        for j in range(3):
            if input_grid[i][j] != 0:
                place_cells(j, i, input_grid[i][j])

    return output_grid
```

**메타데이터**:
- 모델: Mistral-Large-Instruct-2407
- Train 정답률: 5/5 (100%)
- Test 정답률: 1/1 (100%)
- → 이 코드는 **성공한 solution**

**이 task에 대한 다른 2,906개의 solution attempts도 저장됨** (성공/실패 포함)

---

## 7. 데이터 압축 이슈

### Parquet vs HuggingFace Cache

| 형태 | 크기 | 설명 |
|------|------|------|
| **원본 Parquet** | 1.8GB | 고도로 압축된 컬럼 기반 포맷 |
| **HuggingFace Arrow Cache** | 33GB | 압축 해제 + 메타데이터 + 인덱스 |

**주의사항**:
- `load_dataset()` 사용 시 **33GB 추가 디스크 공간** 필요
- Parquet 파일 직접 읽기 권장 (`pyarrow.parquet` 또는 `pandas`)
- Arrow 캐시는 메모리 최적화를 위한 중간 포맷

---

## 8. SOAR 논문 정보

**논문**: Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI
**저자**: Julien Pourcel, Cédric Colas, Pierre-Yves Oudeyer
**학회**: ICML 2025
**HuggingFace**: https://huggingface.co/datasets/julien31/soar_arc_train_5M

**SOAR 모델 (공개됨)**:
- Soar-qwen-7b
- Soar-qwen-14b
- Soar-qwen-32b
- Soar-qwen-72b
- Soar-mistral-123b

---

## 9. 활용 방법

### 기존 ARC 데이터셋 사용 예시

```python
import json

# Load ARC challenge
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

# Get a task
task = challenges['007bbfb7']
train_examples = task['train']
test_input = task['test'][0]['input']

# Your model predicts output
predicted = model.solve(test_input, train_examples)
```

---

### SOAR 데이터셋 사용 예시

```python
import pyarrow.parquet as pq
import pandas as pd

# Load SOAR data
table = pq.read_table('/data/m-soar/soar_arc_train_5M/train_part_0.parquet')
df = table.to_pandas()

# Filter successful solutions for task 007bbfb7
task_data = df[df['task_id'] == '007bbfb7']
successful = task_data[
    task_data['correct_train_input'].apply(lambda x: all(x)) &
    task_data['correct_test_input'].apply(lambda x: all(x))
]

# Extract successful code
for idx, row in successful.iterrows():
    print(f"Model: {row['model']}")
    print(f"Code:\n{row['code']}\n")
```

---

## 10. 요약

| 관점 | 기존 ARC | SOAR |
|------|----------|------|
| **본질** | 문제집 | 풀이집 |
| **역할** | 평가 데이터 | 학습 데이터 |
| **크기** | 작음 (50MB) | 큼 (1.8GB) |
| **다양성** | 400 문제 | 492만 solutions |
| **활용** | 모델 테스트 | 모델 훈련 |

**핵심 차이**:
- ARC는 **"무엇을 풀어야 하는가"** (문제 정의)
- SOAR는 **"어떻게 풀 수 있는가"** (해법 학습)

SOAR 데이터셋은 기존 ARC 문제에 대해 **다양한 모델들이 시도한 수백만 개의 코드 솔루션**을 제공하여, LLM이 **스스로 학습하고 개선**할 수 있도록 설계되었습니다.
