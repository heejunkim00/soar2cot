# SOAR2COT: Natural Language Instructions from Code Solutions

## Project Overview

SOAR2COT is a pipeline that transforms Python code solutions from the ARC (Abstraction and Reasoning Corpus) challenge into natural language instructions and uses them to predict test cases.

**Core Objective**: Convert code-based solutions from the SOAR dataset into human-readable instructions to evaluate LLMs' ability to solve ARC problems using natural language only.

---

## Dataset Scale

- **SOAR Solutions**: 4,926,487 solutions (400 unique tasks)
- **ARC Validation**: 400 tasks
- **Current Processing**:
  - GPT-OSS 20B: 70 tasks completed (469 instructions generated)
  - Qwen3-32B: In progress

---

## Pipeline Architecture

### Whole Process

```
Step 1-3: Instruction Generation and Refinement

Step 1: Intuitive Instructions
  ├─ Input: Training examples + Python code
  ├─ Output: Initial natural language instructions
  └─ Model: Local vLLM (GPT-OSS 20B / Qwen3-32B)

Step 2: Structured Instructions
  ├─ Input: Step 1 instructions + training examples
  ├─ Output: Refined structured instructions
  └─ Scoring: Calculate training accuracy

Step 3: Revision (Optional)
  ├─ Input: Best instructions from Step 2
  ├─ Output: Revised instructions
  └─ Goal: Improve training accuracy to 100%

Step 4: Final Answer Generation

Final Step: Test Predictions
  ├─ Input: Best instructions (score=1.0) + test inputs
  ├─ Process: Generate 2 diverse attempts per test
  ├─ Output: Test output grids
  └─ Scoring: Compare with ground truth (SOAR predicted outputs)
```

### Data Flow

```
SOAR Data
  └─> Challenge Creation
       └─> Step 1-3: Instruction Generation/Refinement
            └─> InstructionsScore (training accuracy)
                 └─> Step 4: Final Answer Generation (score=1.0 only)
                      └─> Guess Object (2 attempts)
                           └─> Answer Comparison (SOAR predicted_test_output)
                                └─> Database Storage
                                     ├─ instructions table
                                     ├─ guess table
                                     └─ scores recording
```

---

## Database Schema

### Instructions Table
```sql
CREATE TABLE instructions (
    id UUID PRIMARY KEY,
    task_id VARCHAR,              -- ARC task identifier
    soar_code TEXT,               -- Python code from SOAR
    instructions TEXT,            -- Generated natural language
    score FLOAT,                  -- Training accuracy (0.0-1.0)
    soar_source_model VARCHAR,    -- SOAR model used
    soar_generation INT,          -- SOAR generation number
    is_hindsight BOOLEAN,         -- Hindsight mode flag
    created_at TIMESTAMP
);
```

### Guess Table
```sql
CREATE TABLE guess (
    id UUID PRIMARY KEY,
    instructions_score_id UUID,   -- FK to instructions
    attempt_number INT,           -- 1 or 2
    avg_score FLOAT,              -- Test accuracy (0.0-1.0)
    scores JSONB,                 -- Per-test-case scores
    model VARCHAR,                -- Model used for prediction
    created_at TIMESTAMP
);
```

### Final Output Structure
```python
Index(['task_id', 'python_code', 'instructions', 'training_score',
       'best_test_score', 'soar_source_model', 'soar_generation',
       'soar_round_index', 'is_hindsight', 'num_guesses', 'created_at'],
      dtype='object')
```

---

## Experimental Results (GPT-OSS 20B)

### Training Performance
- **Total Instructions**: 469 generated
- **Unique Tasks**: 70 completed
- **Average Training Score**: 0.797
- **Perfect Training Score**: 41 tasks (58.6%)

### Test Performance
- **Total Test Attempts**: 138 guesses (2 per task)
- **Average Test Score**: 0.407
- **Perfect Test Cases**: 25 tasks with score=1.0

### Key Insights
1. **Training vs Test Gap**: 0.797 → 0.407 (approximately 49% decrease)
   - Instructions tend to overfit to training examples

2. **Perfect Cases Analysis**:
   - 41 tasks achieved training score 1.0
   - Of these, 25 achieved test score 1.0 (61% generalization)

3. **Model Behavior**:
   - Higher training accuracy tends to correlate with higher test accuracy
   - However, not a perfect correlation (some tasks with training 1.0 fail on test)

---

## Multi-Model Experiment Setup

### Infrastructure
```
soar2cot/
├── data/
│   ├── gpt-oss/              # GPT-OSS 20B experiments
│   │   └── progress.json     # 69 tasks completed
│   ├── qwen3/                # Qwen3-32B experiments
│   │   └── progress.json     # In progress
│   └── perfect_instructions_stats.txt
├── src/
│   ├── configs/
│   │   ├── oss_configs.py    # GPT-OSS configuration
│   │   └── qwen3_configs.py  # Qwen3 configuration
│   └── run.py                # Main pipeline
└── scripts/
    └── run_with_model.sh     # Unified execution script
```

### Database Separation
- **GPT-OSS**: `postgresql://localhost/arc`
- **Qwen3**: `postgresql://localhost/arc_qwen3`

### Model Configurations

#### GPT-OSS 20B
- Server: vLLM on GPUs 0,1,2,3 (tensor parallel 4-way)
- Max tokens: 4096
- Timeout: 300 seconds per step
- Concurrent tasks: 4

#### Qwen3-32B
- Server: vLLM on GPUs 0,1,2,3 (tensor parallel 4-way)
- Max tokens: 8192
- Timeout: 300 seconds per step
- Concurrent tasks: 4

---

## Current Status (2025-11-13)

### Completed
✅ GPT-OSS 20B baseline experiment (70/400 tasks)
✅ Multi-model infrastructure setup
✅ Database schema and storage system
✅ Perfect instructions extraction (25 cases)
✅ Qwen3-32B server initialization

### In Progress
🔄 Qwen3-32B full validation run (400 tasks)
🔄 Comparative analysis between models

### Next Steps
1. Complete Qwen3-32B validation run
2. Comparative analysis: GPT-OSS vs Qwen3
3. Error pattern analysis for failed cases
4. Prompt engineering improvements based on failure modes
5. Scale to full training set (400 tasks → larger dataset)

---

## Technical Challenges Resolved

### 1. Port Conflict Issue
- **Problem**: Multiple vLLM servers competing for port 8000
- **Solution**: Proper cleanup of existing servers before restart
- **Impact**: Prevented silent failures in inference requests

### 2. Output Buffering
- **Problem**: Python buffering prevented real-time logging
- **Solution**: Added `PYTHONUNBUFFERED=1` and `-u` flag
- **Impact**: Improved monitoring and debugging capabilities

### 3. Database Permissions
- **Problem**: User lacked CREATEDB privilege
- **Solution**: Used postgres superuser for database creation
- **Impact**: Enabled multi-model database separation

### 4. Module Import Path
- **Problem**: `python src/run.py` failed with ModuleNotFoundError
- **Solution**: Changed to `python -m src.run`
- **Impact**: Proper Python path resolution

---

## Performance Metrics

### Resource Utilization
- **GPU Memory**: ~15 GiB per GPU (4 GPUs total)
- **Model Loading Time**: ~2 minutes (including CUDA graph capture)
- **Average Task Processing**: 5-10 minutes per task
- **Concurrent Processing**: 4 tasks in parallel

### Throughput
- **GPT-OSS 20B**: 70 tasks completed
- **Expected Completion Time**: ~33 hours for 400 tasks (at 4 concurrent)

---

## Contact & Repository

**Project Lead**: Heejun Kim
**Environment**: AWS EC2 (GPUs 0-7, using 0-3 for experiments)
**Code Base**: `/data/hjkim/soar2cot`

---

## Appendix: Key Configuration

### Environment Variables
```bash
# Model selection
MODEL_CONFIG=qwen3  # or gpt-oss

# Progress tracking
PROGRESS_FILE=/data/hjkim/soar2cot/data/qwen3/progress.json

# Database connection
NEON_DSN_GPT_OSS=postgresql://hjkim:***@localhost/arc
NEON_DSN_QWEN3=postgresql://hjkim:***@localhost/arc_qwen3
```

### Execution Command
```bash
# Run with specific model
./scripts/run_with_model.sh qwen3

# Monitor progress
tail -f logs/run_qwen3_*.log
watch -n 5 "cat data/qwen3/progress.json | jq '.total_completed'"
```

---

*Document generated: 2025-11-13*
*Last updated: Based on real-time experimental data*
