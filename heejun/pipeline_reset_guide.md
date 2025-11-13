# 파이프라인 초기화 가이드

## 개요

파이프라인을 **처음부터 다시 실행**하고 싶을 때 초기화하는 방법입니다.

---

## 초기화가 필요한 경우

### 1. 완전히 새로 시작하고 싶을 때
- 테스트 실행 결과를 지우고 본격적으로 시작
- 코드 수정 후 전체 재실행

### 2. 데이터 오염이 의심될 때
- 버그로 인해 잘못된 데이터가 저장됨
- DB에 중복 데이터가 쌓임

### 3. Config 변경 후
- 다른 모델로 전환
- 다른 프롬프트로 전환

---

## 초기화 대상

### 1. Progress Tracker
**파일**: `/data/hjkim/soar2cot/data/progress.json`

**역할**:
- 완료된 (task_id, round_index, data_type) 추적
- 재시작 시 완료된 작업 건너뛰기

**초기화 필요 이유**:
- progress.json이 있으면 → 완료된 작업 건너뜀
- progress.json이 없으면 → 모든 작업 처리

### 2. Database Tables

#### `instructions` 테이블
```sql
CREATE TABLE instructions (
    id TEXT PRIMARY KEY,
    instructions TEXT NOT NULL,      -- 생성된 instruction
    model TEXT NOT NULL,              -- 사용한 모델
    example_scores JSONB NOT NULL,    -- 각 training example별 점수
    score DOUBLE PRECISION NOT NULL,  -- 평균 training score
    task_id TEXT NOT NULL,
    task_hash TEXT NOT NULL,
    step JSONB NOT NULL,
    soar_code TEXT,                   -- Python 코드 (SOAR)
    soar_source_model TEXT,           -- SOAR 코드 생성 모델
    soar_generation INTEGER,          -- SOAR generation
    soar_round_index INTEGER,         -- Round 번호
    is_hindsight BOOLEAN,             -- Hindsight 여부
    created_at TIMESTAMP
);
```

**저장 시점**: instruction 생성 및 training 평가 후

#### `guess` 테이블
```sql
CREATE TABLE guess (
    id TEXT PRIMARY KEY,
    grids JSONB NOT NULL,                    -- 생성한 test output grids
    instructions_score_id TEXT NOT NULL,     -- instructions 테이블의 id (외래키)
    model TEXT NOT NULL,
    avg_score DOUBLE PRECISION NOT NULL,     -- Test score 평균
    scores JSONB NOT NULL,                   -- 각 test input별 점수
    created_at TIMESTAMP,
    FOREIGN KEY (instructions_score_id) REFERENCES instructions(id)
);
```

**저장 시점**: test output 생성 및 평가 후

**초기화 필요 이유**:
- 같은 task를 두 번 처리하면 → DB에 중복 레코드 생성
- progress.json과 DB가 동기화되지 않으면 → 데이터 불일치

### 3. Log Directories (선택)

**위치**: `/data/hjkim/soar2cot/logs/run_YYYYMMDD_HHMMSS/`

**내용**:
- LLM prompts (첫 5개)
- LLM responses (첫 5개)
- Truncated responses (첫 10개)

**초기화 선택 이유**:
- 로그는 디버깅용이므로 필수 아님
- 하지만 용량을 차지하므로 정리 가능

---

## 초기화 방법

### 방법 1: 자동 스크립트 (권장)

```bash
# 스크립트 실행
cd /data/hjkim/soar2cot
./scripts/reset_pipeline.sh
```

**동작**:
1. progress.json 삭제 확인
2. Database 테이블 초기화 (TRUNCATE)
3. 로그 디렉토리 삭제 (선택)

**출력 예시**:
```
==========================================
Pipeline Reset Script
==========================================

This script will:
  1. Delete progress.json (task completion tracking)
  2. Truncate database tables (instructions and guess)
  3. Clean up old log directories (optional)

WARNING: This will permanently delete all progress and results!

Are you sure you want to reset the pipeline? (yes/no): yes

[1/3] Deleting progress.json...
✓ Deleted: /data/hjkim/soar2cot/data/progress.json

[2/3] Truncating database tables...
Database: arc @ localhost
User: hjkim
✓ Database tables truncated

[3/3] Cleaning up old logs...
Do you want to delete old log directories? (yes/no): yes
Found 1 log directories
Deleting old logs...
✓ Old logs deleted

==========================================
Pipeline Reset Complete!
==========================================

You can now run the pipeline from scratch:
  python -m src.run
```

### 방법 2: 수동 초기화

#### Step 1: progress.json 삭제
```bash
rm /data/hjkim/soar2cot/data/progress.json
```

#### Step 2: Database 초기화
```bash
# PostgreSQL 접속
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc

# 테이블 초기화
TRUNCATE TABLE guess CASCADE;       -- guess 먼저 (외래키 때문)
TRUNCATE TABLE instructions CASCADE;

-- 확인
SELECT COUNT(*) FROM instructions;  -- 0이어야 함
SELECT COUNT(*) FROM guess;         -- 0이어야 함

-- 종료
\q
```

#### Step 3: 로그 정리 (선택)
```bash
# 모든 이전 로그 삭제
rm -rf /data/hjkim/soar2cot/logs/run_*

# 또는 특정 날짜만 삭제
rm -rf /data/hjkim/soar2cot/logs/run_20251112_*
```

---

## 주의사항

### 1. 백업 먼저!

중요한 데이터가 있다면 초기화 전에 백업:

```bash
# progress.json 백업
cp /data/hjkim/soar2cot/data/progress.json \
   /data/hjkim/soar2cot/data/progress_backup_$(date +%Y%m%d_%H%M%S).json

# DB 백업
PGPASSWORD=hjkim123 pg_dump -h localhost -U hjkim -d arc \
   -t instructions -t guess \
   > /data/hjkim/soar2cot/backups/db_backup_$(date +%Y%m%d_%H%M%S).sql
```

### 2. 초기화는 되돌릴 수 없음

- `TRUNCATE` 명령은 **복구 불가능**
- 실행 전에 반드시 확인!

### 3. 외래키 제약

`guess` 테이블은 `instructions` 테이블을 참조하므로:
- `guess` 먼저 삭제/초기화
- 그 다음 `instructions` 초기화
- `CASCADE` 옵션으로 자동 처리 가능

### 4. 실행 중인 파이프라인 중단

초기화 전에 파이프라인이 실행 중이라면:
```bash
# Ctrl+C로 중단
# 또는 프로세스 찾아서 종료
ps aux | grep "python -m src.run"
kill <PID>
```

---

## 부분 초기화

### 특정 task만 재처리하고 싶을 때

progress.json에서 해당 task 항목만 삭제:

```bash
# progress.json 편집
vim /data/hjkim/soar2cot/data/progress.json

# 예시: task_id "007bbfb7"를 다시 처리하려면
# completed 리스트에서 해당 항목 삭제:
{
  "completed": [
    {"task_id": "007bbfb7", "round_index": 0, "data_type": "original"},  # ← 삭제
    {"task_id": "00d62c1b", "round_index": 0, "data_type": "original"}
  ],
  "total_completed": 2
}

# 저장 후 재실행
python -m src.run
# → 007bbfb7만 다시 처리됨
```

**주의**: DB에서도 해당 task 데이터를 삭제해야 중복 방지:

```sql
-- instructions 삭제 (CASCADE로 guess도 자동 삭제)
DELETE FROM instructions WHERE task_id = '007bbfb7' AND soar_round_index = 0;
```

### 특정 round만 재처리

```bash
# Round 0만 다시 처리
# progress.json에서 round_index=0 항목 모두 삭제

# DB에서도 삭제
DELETE FROM instructions WHERE soar_round_index = 0;
```

### Hindsight만 재처리

```bash
# is_hindsight=True 항목만 삭제
# progress.json에서 data_type="hindsight" 삭제

# DB에서도 삭제
DELETE FROM instructions WHERE is_hindsight = TRUE;
```

---

## 초기화 후 확인

### 1. progress.json 확인
```bash
ls -l /data/hjkim/soar2cot/data/progress.json
# 출력: No such file or directory ✓
```

### 2. DB 확인
```bash
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c "SELECT COUNT(*) FROM instructions; SELECT COUNT(*) FROM guess;"
# 출력:
# count: 0 ✓
# count: 0 ✓
```

### 3. 파이프라인 실행
```bash
python -m src.run

# 로그 확인:
# No progress file found, starting fresh ✓
# Loading original SOAR data ✓
# Processing round 0 samples=... ✓
```

---

## 스크립트 위치

- **초기화 스크립트**: `/data/hjkim/soar2cot/scripts/reset_pipeline.sh`
- **이 가이드**: `/data/hjkim/soar2cot/heejun/pipeline_reset_guide.md`

---

## 요약

### ✅ 완전 초기화 (처음부터 다시)
```bash
./scripts/reset_pipeline.sh
```

### ✅ 수동 초기화
```bash
# 1. Progress 삭제
rm /data/hjkim/soar2cot/data/progress.json

# 2. DB 초기화
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc << EOF
TRUNCATE TABLE guess CASCADE;
TRUNCATE TABLE instructions CASCADE;
EOF

# 3. 확인
python -m src.run
```

### ✅ 부분 초기화 (특정 task만)
```bash
# 1. progress.json 편집 (해당 task 항목 삭제)
vim /data/hjkim/soar2cot/data/progress.json

# 2. DB에서 해당 task 삭제
PGPASSWORD=hjkim123 psql -h localhost -U hjkim -d arc -c \
  "DELETE FROM instructions WHERE task_id = 'TASK_ID';"

# 3. 재실행
python -m src.run
```
