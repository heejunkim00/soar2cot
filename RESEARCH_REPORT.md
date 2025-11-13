# SOAR2COT: From Code Solutions to Natural Language Instructions

**Research Progress Report**
Heejun Kim
November 13, 2025

---

## Executive Summary

This report presents our progress on SOAR2COT, a project that transforms code-based solutions from the SOAR dataset into natural language instructions for solving ARC (Abstraction and Reasoning Corpus) challenges. We have developed a multi-stage pipeline that generates, refines, and evaluates natural language instructions, achieving perfect test accuracy (1.0) on 25 out of 70 completed tasks using GPT-OSS 20B.

**Key Results:**
- **Training Accuracy**: 0.797 average (41 tasks with perfect 1.0 score)
- **Test Accuracy**: 0.407 average (25 tasks with perfect 1.0 score)
- **Generalization Rate**: 61% of perfect training cases successfully generalize to test cases
- **Scale**: Processing 4.9M SOAR solutions across 400 ARC validation tasks

---

## 1. Research Motivation

The ARC challenge requires human-like abstract reasoning, and while SOAR has generated millions of Python code solutions, these remain opaque to interpretability analysis. Our objective is to bridge this gap by:

1. Converting executable code into interpretable natural language instructions
2. Evaluating whether language-based reasoning can match code-based performance
3. Identifying patterns in successful instruction generation for reasoning tasks
4. Building a foundation for interpretable AI reasoning systems

---

## 2. Methodology

### 2.1 Pipeline Architecture

Our pipeline consists of four primary stages:

#### **Stage 1: Intuitive Instruction Generation**
- **Input**: Training examples (input/output grids) + Python code solution
- **Process**: LLM generates initial natural language description
- **Output**: Human-readable instructions explaining the transformation pattern

#### **Stage 2: Structured Instruction Refinement**
- **Input**: Stage 1 instructions + training examples
- **Process**: LLM refines instructions into structured, step-by-step format
- **Evaluation**: Calculate training accuracy by generating outputs and comparing with ground truth
- **Output**: Refined instructions with training accuracy score

#### **Stage 3: Iterative Revision (Optional)**
- **Input**: Best-performing instructions from Stage 2
- **Process**: Further refinement targeting 100% training accuracy
- **Termination**: Continue until score = 1.0 or max iterations reached

#### **Stage 4: Test Prediction**
- **Input**: Instructions with perfect training score (1.0) + test input grids
- **Process**: Generate 2 diverse prediction attempts per test case
- **Evaluation**: Compare predictions with SOAR ground truth outputs
- **Output**: Final test accuracy score

### 2.2 Data Flow

```
SOAR Dataset (4.9M solutions, 400 tasks)
  ↓
Challenge Creation
  ↓
Instruction Generation Pipeline (Stages 1-3)
  ↓
Instruction Scoring (training accuracy)
  ↓
Test Prediction (score = 1.0 only)
  ↓
Evaluation (compare with SOAR predictions)
  ↓
PostgreSQL Database Storage
```

### 2.3 Database Schema

We maintain two primary tables:

**Instructions Table:**
```sql
- id: UUID (primary key)
- task_id: ARC task identifier
- soar_code: Original Python solution
- instructions: Generated natural language
- score: Training accuracy (0.0-1.0)
- soar_source_model: SOAR model identifier
- soar_generation: Generation number
- is_hindsight: Boolean flag
- created_at: Timestamp
```

**Guess Table:**
```sql
- id: UUID (primary key)
- instructions_score_id: Foreign key to instructions
- attempt_number: 1 or 2
- avg_score: Test accuracy (0.0-1.0)
- scores: JSONB (per-test-case scores)
- model: Model identifier
- created_at: Timestamp
```

---

## 3. Experimental Setup

### 3.1 Models

We are conducting experiments with two large language models:

#### **GPT-OSS 20B**
- Deployment: vLLM server with 4-way tensor parallelism (GPUs 0-3)
- Context length: 4096 tokens
- Timeout: 300 seconds per step
- Concurrent processing: 4 tasks
- Status: **70/400 tasks completed**

#### **Qwen3-32B**
- Deployment: vLLM server with 4-way tensor parallelism (GPUs 0-3)
- Context length: 8192 tokens
- Timeout: 300 seconds per step
- Concurrent processing: 4 tasks
- Status: **Currently running**

### 3.2 Infrastructure

- **Environment**: AWS EC2 with 8 GPUs (4 allocated per experiment)
- **Storage**: Separate PostgreSQL databases per model
- **Progress Tracking**: JSON-based checkpoint system with resume capability
- **Logging**: Comprehensive logging for debugging and analysis

---

## 4. Results and Analysis

### 4.1 Quantitative Results (GPT-OSS 20B)

| Metric | Value |
|--------|-------|
| Tasks Completed | 70 / 400 |
| Instructions Generated | 469 |
| Avg Training Score | 0.797 |
| Perfect Training Cases | 41 (58.6%) |
| Avg Test Score | 0.407 |
| Perfect Test Cases | 25 (35.7%) |
| Total Test Attempts | 138 (2 per task) |

### 4.2 Key Findings

#### **Finding 1: Significant Train-Test Gap**
Training accuracy (0.797) drops to test accuracy (0.407), representing a 49% decrease. This suggests:
- Instructions tend to overfit to training examples
- Generalization from natural language descriptions is challenging
- Some transformation patterns are not fully captured in instructions

#### **Finding 2: Strong Generalization in Perfect Cases**
Among 41 tasks with perfect training scores (1.0):
- 25 tasks (61%) maintained perfect test scores
- 16 tasks (39%) failed to generalize despite perfect training fit
- This indicates that training accuracy is a necessary but not sufficient condition

#### **Finding 3: Instruction Quality Varies Significantly**
- Best performing tasks: Clear, unambiguous transformation rules
- Poorly performing tasks: Complex spatial reasoning or multi-step transformations
- Middle-tier tasks: Partial success suggests incomplete rule capture

### 4.3 Perfect Instruction Cases

We have extracted 25 high-quality instruction sets that achieve perfect scores on both training and test cases. These represent:
- Clear, actionable transformation rules
- Successful abstraction from code to language
- Potential gold-standard examples for future prompt engineering

**Example Task Distribution:**
```
29c11459: train=1.000, test=1.000
25d8a9c8: train=1.000, test=1.000
25ff71a9: train=1.000, test=1.000
23b5c85d: train=1.000, test=1.000
... (21 more perfect cases)
```

---

## 5. Technical Implementation

### 5.1 Multi-Model Infrastructure

We developed a flexible infrastructure supporting multiple models with isolated environments:

```bash
# Directory structure
soar2cot/
├── data/
│   ├── gpt-oss/          # Model-specific data
│   └── qwen3/            # Separate progress tracking
├── src/
│   └── configs/          # Per-model configurations
└── scripts/
    └── run_with_model.sh # Unified execution script
```

### 5.2 Progress Tracking and Resume

Our system implements robust progress tracking:
- JSON-based checkpoint files
- Database-backed state recovery
- Graceful handling of interruptions
- Independent progress per model

### 5.3 Performance Optimization

- **Parallel Processing**: 4 concurrent tasks per model
- **Tensor Parallelism**: Model sharding across 4 GPUs
- **CUDA Graph Optimization**: Pre-recorded GPU operations
- **Resource Isolation**: Separate GPU allocation (0-3 vs 4-7)

---

## 6. Challenges and Solutions

### 6.1 Port Conflict Issue
**Problem**: Multiple vLLM servers competing for port 8000 caused silent failures
**Solution**: Proper cleanup of existing servers before restart
**Impact**: Prevented inference request failures

### 6.2 Output Buffering
**Problem**: Python buffering prevented real-time logging
**Solution**: Added `PYTHONUNBUFFERED=1` environment variable
**Impact**: Improved monitoring and debugging capabilities

### 6.3 Database Permissions
**Problem**: User lacked database creation privileges
**Solution**: Used postgres superuser for database initialization
**Impact**: Enabled multi-model database separation

### 6.4 Module Import Resolution
**Problem**: Direct script execution failed with import errors
**Solution**: Changed from `python src/run.py` to `python -m src.run`
**Impact**: Proper Python path resolution

---

## 7. Next Steps

### 7.1 Immediate Priorities

1. **Complete Qwen3-32B Evaluation** (in progress)
   - Finish all 400 validation tasks
   - Compare with GPT-OSS 20B results
   - Analyze model-specific strengths

2. **Comparative Model Analysis**
   - Identify which tasks favor which model
   - Analyze instruction generation quality differences
   - Determine optimal model selection strategy

3. **Error Pattern Analysis**
   - Categorize failure modes
   - Identify common characteristics in failed cases
   - Extract insights for prompt engineering

### 7.2 Medium-Term Goals

4. **Prompt Engineering Improvements**
   - Refine instruction generation prompts based on failure analysis
   - Incorporate successful examples into few-shot prompts
   - Test structured output formats

5. **Scale Expansion**
   - Extend beyond 400 validation tasks
   - Process larger subsets of SOAR dataset
   - Evaluate on additional test sets

6. **Instruction Quality Metrics**
   - Develop automated quality assessment
   - Measure instruction clarity and completeness
   - Create instruction evaluation framework

### 7.3 Long-Term Research Directions

7. **Human Evaluation Study**
   - Recruit human evaluators to assess instruction quality
   - Compare human understanding vs model performance
   - Validate interpretability claims

8. **Instruction Refinement Techniques**
   - Explore iterative human-in-the-loop refinement
   - Test automated instruction improvement strategies
   - Investigate ensemble instruction methods

9. **Cross-Domain Generalization**
   - Test instructions on related reasoning tasks
   - Evaluate transfer learning potential
   - Explore zero-shot generalization

---

## 8. Resource Utilization

### 8.1 Computational Resources

- **GPU Memory**: ~15 GiB per GPU (60 GiB total per model)
- **Model Loading Time**: ~2 minutes (including CUDA graph capture)
- **Processing Time**: 5-10 minutes per task
- **Expected Runtime**: ~33 hours for 400 tasks (4 concurrent)

### 8.2 Storage Requirements

- **SOAR Dataset**: 4.9M solutions (~5 GB compressed)
- **Database Size**: ~50 MB per 70 tasks
- **Log Files**: ~100 MB per full run
- **Checkpoint Files**: Minimal (<1 MB per model)

---

## 9. Reproducibility

### 9.1 Repository

**Code**: https://github.com/heejunkim00/soar2cot
**Branch**: `main`
**Commit**: fc5e88c

### 9.2 Key Files

- `src/run.py`: Main pipeline orchestration
- `src/configs/oss_configs.py`: GPT-OSS configuration
- `src/configs/qwen3_configs.py`: Qwen3 configuration
- `scripts/run_with_model.sh`: Execution script
- `start_vllm_qwen3.sh`: Server initialization

### 9.3 Execution

```bash
# Start vLLM server
./start_vllm_qwen3.sh

# Run experiment
source .env
./scripts/run_with_model.sh qwen3

# Monitor progress
tail -f logs/run_qwen3_*.log
```

---

## 10. Conclusion

Our SOAR2COT pipeline demonstrates promising results in converting code-based ARC solutions to natural language instructions, achieving perfect generalization on 25 tasks (35.7% of completed tasks). The 61% generalization rate among perfect training cases suggests that high-quality instructions can reliably capture abstract reasoning patterns.

The significant train-test gap (0.797 → 0.407) highlights the challenge of generalization in instruction-based reasoning, presenting opportunities for future research in prompt engineering, instruction refinement, and model selection strategies.

With Qwen3-32B experiments currently underway, we expect to gain deeper insights into model-specific capabilities and identify optimal approaches for different task categories. The infrastructure we have built enables rapid experimentation with additional models and scaling to larger datasets.

---

## Appendix: Configuration Summary

### A.1 Environment Variables

```bash
MODEL_CONFIG=qwen3                    # Model selection
PROGRESS_FILE=data/qwen3/progress.json  # Progress tracking
NEON_DSN=postgresql://localhost/arc_qwen3  # Database connection
```

### A.2 Model Parameters

```python
RunConfig(
    final_follow_model=Model.local_qwen3_32b,
    final_follow_times=3,
    max_concurrent_tasks=4,
    steps=[
        Step(times=2, timeout_secs=300, use_diffs=True),
        Step(times=2, timeout_secs=300, use_diffs=True),
        Step(times=3, timeout_secs=300, use_diffs=True),
        StepRevisionPool(top_scores_used=3, times=2),
    ]
)
```

---

**Contact Information**
Heejun Kim
Georgia Institute of Technology
Research supervised by Prof. Vijay

**Document Version**: 1.0
**Last Updated**: November 13, 2025
