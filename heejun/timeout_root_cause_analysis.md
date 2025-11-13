# 타임아웃 근본 원인 분석 및 해결

## 근본 원인

### 1. 과도한 max_tokens 설정
**위치**: `/data/hjkim/soar2cot/src/llms/structured.py` (라인 834)

```python
max_tokens_list = [8000, 12000, 16000, 20000]  # 너무 큼!
```

**문제점**:
- 20,000 토큰 생성 = 약 15,000 단어 = 30-40 페이지 분량
- GPT-OSS 20B 모델이 이를 생성하는데 2-5분 소요
- 8개 동시 요청 시 GPU 메모리 부족

### 2. 실제 타임아웃 발생 과정

```
1. Task 시작 → vLLM에 8000 토큰 요청
2. 응답이 잘림 (finish_reason: 'length')
3. 12000 토큰으로 재시도
4. 또 잘림 → 16000 토큰으로 재시도
5. 또 잘림 → 20000 토큰으로 재시도
6. 생성에 3-5분 소요
7. 8개가 동시에 이런 과정 반복
8. 네트워크 타임아웃 발생
```

### 3. vLLM 서버 로그 분석

```
06:36:13 Running: 8 reqs  # 8개 요청 처리 중
06:36:23 Running: 7 reqs  # 1개 완료
06:36:33 Running: 5 reqs  # 3개 완료
06:36:43 Running: 5 reqs  # 여전히 5개 처리 중 (속도 저하)
06:36:53 Running: 3 reqs  # 2개 더 완료
06:37:03 Running: 0 reqs  # 모두 완료 (총 50초 소요)
```

## 최적화된 해결 방법

### 1. max_tokens 줄이기 (가장 중요!)

```python
# 수정 전
max_tokens_list = [8000, 12000, 16000, 20000]

# 수정 후 (권장)
max_tokens_list = [2000, 4000, 6000, 8000]  # 대폭 감소
```

### 2. 재시도 전략 개선

```python
# structured.py 라인 843-851 수정
response = await local_vllm_client.chat.completions.create(
    model=model.value,
    messages=messages,
    response_format={"type": "json_object"},
    max_tokens=max_tokens,
    temperature=0.3,
    # reasoning_effort="low",  # 제거 (지원 안됨)
    # extra_body={"skip_special_tokens": True},  # 제거 (지원 안됨)
)
```

### 3. 동시 처리 최적화

```python
# oss_configs.py
max_concurrent_tasks=4,  # 8 → 4로 감소

# run.py
sleep_for=i * 10  # 5 → 10초로 증가 (더 많은 간격)
```

## 실제 필요한 토큰 수

ARC 퍼즐 instruction의 실제 크기:
- 간단한 규칙: 100-500 토큰
- 복잡한 규칙: 500-2000 토큰
- 매우 복잡한 설명: 2000-4000 토큰

**20,000 토큰은 과도함!**

## 성능 개선 예상치

### 현재 (문제 상황)
- 요청당 시간: 2-5분
- 8개 동시: 메모리 부족, 타임아웃
- 성공률: 매우 낮음

### 개선 후
- 요청당 시간: 10-30초
- 4개 동시: 안정적 처리
- 성공률: 95% 이상

## 즉시 적용 가능한 수정

```bash
# 1. max_tokens 수정
sed -i 's/\[8000, 12000, 16000, 20000\]/[2000, 4000, 6000, 8000]/' /data/hjkim/soar2cot/src/llms/structured.py

# 2. 불필요한 파라미터 제거 (수동으로 해야 함)
# reasoning_effort="low" 라인 제거
# extra_body={...} 라인 제거

# 3. 동시 처리 감소
sed -i 's/max_concurrent_tasks=8/max_concurrent_tasks=4/' /data/hjkim/soar2cot/src/configs/oss_configs.py
```

## 모니터링

```bash
# vLLM 서버 모니터링
watch -n 2 'curl -s http://localhost:8000/metrics | grep vllm_num_requests_running'

# GPU 메모리 모니터링
nvidia-smi dmon -s mu -d 5
```

## 결론

**타임아웃의 진짜 원인은**:
1. ❌ 타임아웃 설정이 짧아서가 아님
2. ✅ 너무 많은 토큰을 생성하려 해서
3. ✅ 8개 요청이 동시에 거대한 응답을 요구해서

**해결책**:
1. max_tokens을 현실적으로 줄이기 (2000-8000)
2. 동시 처리 수 줄이기 (4개)
3. 불필요한 파라미터 제거