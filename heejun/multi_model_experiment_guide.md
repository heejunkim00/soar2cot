# Multi-Model Experiment Guide

SOAR2COT 프레임워크에서 여러 모델(GPT-OSS, Qwen3)로 실험하는 방법

## 구조

### 디렉토리 구조
```
/data/hjkim/soar2cot/
├── data/
│   ├── gpt-oss/
│   │   └── progress.json (69 completed tasks)
│   └── qwen3/
│       └── progress.json (자동 생성)
├── start_vllm_server.sh      # GPT-OSS 서버
├── start_vllm_qwen3.sh        # Qwen3 서버
└── scripts/
    └── run_with_model.sh      # 실험 실행 스크립트
```

### 분리된 요소
- **Progress 파일**: 모델별 디렉토리에 자동 분리
- **Database**: 환경변수로 분리 (수동 설정 필요)
- **Config**: `MODEL_CONFIG` 환경변수로 자동 선택

---

## GPT-OSS 사용 방법

### 1. DB 환경변수 설정
```bash
export NEON_DSN_GPT_OSS="postgresql://user:pass@host/db_name"
# 또는 기존 DB 사용
export NEON_DSN_GPT_OSS="$NEON_DSN"
```

### 2. vLLM 서버 시작/확인
```bash
# 이미 실행 중이면 생략
./start_vllm_server.sh
```

**서버 설정:**
- 모델: `openai/gpt-oss-20b`
- GPU: 0,1,2,3 (tensor parallel 4)
- 포트: 8000
- Reasoning: ❌

### 3. 파이프라인 실행
```bash
./scripts/run_with_model.sh gpt-oss
```

### 결과
- Progress: `/data/hjkim/soar2cot/data/gpt-oss/progress.json` (69개부터 이어서)
- DB: `$NEON_DSN_GPT_OSS`
- Config: `local_gpt_oss_20b_config`
- Log: `logs/run_gpt-oss_YYYYMMDD_HHMMSS.log`

---

## Qwen3 사용 방법

### 1. DB 환경변수 설정
```bash
export NEON_DSN_QWEN3="postgresql://user:pass@host/db_qwen3_new"
```

⚠️ **중요**: Qwen3용 **새로운 DB**를 생성해야 합니다 (GPT-OSS DB와 분리)

### 2. GPT-OSS vLLM 서버 종료
```bash
# 프로세스 확인
ps aux | grep vllm | grep -v grep

# 종료
pkill -f "vllm.*gpt-oss"
# 또는 PID로 종료
kill <PID>
```

### 3. Qwen3 vLLM 서버 시작
```bash
./start_vllm_qwen3.sh
```

**서버 설정:**
- 모델: `Qwen/Qwen3-32B`
- GPU: 0,1,2,3 (tensor parallel 4)
- 포트: 8000
- Reasoning: ✅ (`--enable-reasoning --reasoning-parser deepseek_r1`)
- Max model length: 8192
- GPU memory utilization: 0.9

### 4. 파이프라인 실행
```bash
# 새 터미널에서
./scripts/run_with_model.sh qwen3
```

### 결과
- Progress: `/data/hjkim/soar2cot/data/qwen3/progress.json` (처음부터 시작)
- DB: `$NEON_DSN_QWEN3`
- Config: `local_qwen3_32b_config`
- Log: `logs/run_qwen3_YYYYMMDD_HHMMSS.log`

---

## 비교표

| 항목 | GPT-OSS | Qwen3 |
|------|---------|-------|
| **vLLM 시작** | `./start_vllm_server.sh` | `./start_vllm_qwen3.sh` |
| **실행 명령** | `./scripts/run_with_model.sh gpt-oss` | `./scripts/run_with_model.sh qwen3` |
| **Progress 파일** | `data/gpt-oss/progress.json` | `data/qwen3/progress.json` |
| **DB 환경변수** | `NEON_DSN_GPT_OSS` | `NEON_DSN_QWEN3` |
| **모델 ID** | `openai/gpt-oss-20b` | `Qwen/Qwen3-32B` |
| **파라미터** | 20B | 32.8B |
| **Thinking mode** | ❌ | ✅ |
| **현재 진행** | 69개 완료 | 0개 (새 시작) |
| **Config** | `local_gpt_oss_20b_config` | `local_qwen3_32b_config` |

---

## 전체 실행 예시

### GPT-OSS 실행 (기존 실험 계속)

```bash
# 1. 환경변수 설정
export NEON_DSN_GPT_OSS="postgresql://..."

# 2. 실행 (vLLM 서버가 이미 실행 중)
cd /data/hjkim/soar2cot
./scripts/run_with_model.sh gpt-oss
```

### Qwen3 실행 (새 실험 시작)

```bash
# Terminal 1: vLLM 서버
export NEON_DSN_QWEN3="postgresql://user:pass@host/db_qwen3_new"

# GPT-OSS 서버 종료
pkill -f "vllm.*gpt-oss"

# Qwen3 서버 시작
cd /data/hjkim/soar2cot
./start_vllm_qwen3.sh

# Terminal 2: 파이프라인 실행
export NEON_DSN_QWEN3="postgresql://user:pass@host/db_qwen3_new"
cd /data/hjkim/soar2cot
./scripts/run_with_model.sh qwen3
```

---

## 주의사항

### 1. 포트 충돌
- 두 vLLM 서버가 동시에 8000 포트 사용 불가
- 한 모델만 실행 가능
- 전환 시 기존 서버 종료 필수

### 2. DB 분리
- **반드시 별도 DB 사용**
- 같은 DB 사용 시 데이터 충돌
- GPT-OSS: 기존 DB (69개 완료 데이터)
- Qwen3: 새 DB (깨끗한 상태)

### 3. Progress 파일
- 자동으로 모델별 디렉토리에 분리
- 수동 관리 불필요
- 스크립트가 `PROGRESS_FILE` 환경변수 자동 설정

### 4. GPU 메모리
- GPT-OSS 20B: ~125GB (4 GPUs)
- Qwen3-32B: ~85GB (4 GPUs, tensor parallel)
- H200 140GB x 4 = 충분

---

## 환경변수 정리

### 필수 설정
```bash
# GPT-OSS 실험
export NEON_DSN_GPT_OSS="postgresql://..."

# Qwen3 실험
export NEON_DSN_QWEN3="postgresql://..."
```

### 자동 설정 (스크립트가 처리)
```bash
# run_with_model.sh가 자동 설정
export PROGRESS_FILE="..."      # 모델별 progress.json 경로
export MODEL_CONFIG="..."       # 모델 config 선택 (gpt-oss 또는 qwen3)
export NEON_DSN="..."          # 선택된 모델의 DB
```

---

## 트러블슈팅

### vLLM 서버가 종료되지 않을 때
```bash
# 강제 종료
pkill -9 -f vllm

# 프로세스 확인
ps aux | grep vllm
```

### DB 연결 오류
```bash
# 환경변수 확인
echo $NEON_DSN_GPT_OSS
echo $NEON_DSN_QWEN3

# DB 연결 테스트
psql "$NEON_DSN_GPT_OSS" -c "SELECT 1"
```

### Progress 파일 확인
```bash
# 현재 진행 상황 확인
cat /data/hjkim/soar2cot/data/gpt-oss/progress.json | jq '.total_completed'
cat /data/hjkim/soar2cot/data/qwen3/progress.json | jq '.total_completed'
```

### 로그 확인
```bash
# 최신 로그
ls -lht /data/hjkim/soar2cot/logs/run_*.log | head -5

# 실시간 로그 확인
tail -f /data/hjkim/soar2cot/logs/run_qwen3_*.log
```

---

## 코드 수정 사항 (참고)

### 추가된 파일
- `src/llms/models.py`: `local_qwen3_32b` 모델 추가
- `src/configs/qwen3_configs.py`: Qwen3 config 정의
- `start_vllm_qwen3.sh`: Qwen3 vLLM 서버 스크립트
- `scripts/run_with_model.sh`: 통합 실험 실행 스크립트

### 수정된 파일
- `src/run.py` (line 1413-1422): `MODEL_CONFIG` 환경변수로 config 선택
- `src/progress_tracker.py` (line 17-19): `PROGRESS_FILE` 환경변수 지원

### 환경변수 처리 흐름
```
run_with_model.sh
  ↓ export PROGRESS_FILE
  ↓ export MODEL_CONFIG
  ↓ export NEON_DSN
src/progress_tracker.py
  ↓ 읽기: PROGRESS_FILE
src/run.py
  ↓ 읽기: MODEL_CONFIG
  ↓ config 선택: gpt-oss 또는 qwen3
```
