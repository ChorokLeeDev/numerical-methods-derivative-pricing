# 왜 ML이 필요한가?

## TL;DR

전통적 방법의 문제:

1. **Alpha Decay**: 팩터가 논문에 발표되면 모두가 사용 → 수익 감소
2. **Crowding**: "저 PBR 사자" → 가격 상승 → 더 이상 싸지 않음
3. **Estimation Error**: Mean-Variance가 오차를 증폭

ML이 해결하는 것:

| 문제 | ML 해결책 |
|------|-----------|
| 선형 모델의 한계 | 비선형 관계 포착 (Gradient Boosting, NN) |
| 정적 팩터 가중치 | 시장 상태에 따른 동적 조절 |
| "예측했다" vs "얼마나 확신하나" | 불확실성 정량화 (Bayesian 연결) |

## 3주차 구현 로드맵

**A → B → C 순서로 진행**

| 단계 | 방법론 | 난이도 | 시간 | 특징 |
|------|--------|--------|------|------|
| A | LightGBM + Quantile Regression | 쉬움 | 1-2일 | 실용적, 면접 설명 쉬움 |
| B | LSTM + MC Dropout | 중간 | 3-5일 | 논문 연결, Bayesian |
| C | Temporal Fusion Transformer | 어려움 | 5-7일 | 최신, 해석+불확실성 |

---

## 전통적 방법의 한계

### 1. 팩터 투자의 문제

**알파 decay (Alpha Decay)**

```
1992: Fama-French 3팩터 발표 → 학계/업계 적용
2000s: 모멘텀, 퀄리티 팩터 추가
2010s: "팩터 투자"가 상식이 됨
2020s: 모두가 같은 팩터 사용 → 수익 감소
```

논문에서 발견된 팩터가 실제로 작동하는 이유:
1. **위험 프리미엄**: 진짜 위험에 대한 보상 (사라지지 않음)
2. **행동 편향**: 투자자 심리의 비합리성 (천천히 교정됨)
3. **구조적 제약**: 기관의 규제/제약 (변하지만 새로운 것이 생김)

문제는 **2번과 3번**이 시간이 지나면서 arbitrage out 된다는 것.

**팩터 crowding**
```
모두가 "저 PBR 주식 사야지" → 저 PBR 주식 가격 상승 → 더 이상 싸지 않음
```

### 2. Mean-Variance의 문제

**Estimation Error Maximizer**

Markowitz가 "가장 최적"이라고 한 포트폴리오가 실제로는 "추정 오차에 가장 민감한" 포트폴리오.

```
입력: μ (기대수익률), Σ (공분산)
둘 다 과거 데이터로 추정 → 오차 불가피
최적화가 오차를 증폭시킴
```

**Black-Litterman의 한계**

BL은 이 문제를 완화하지만:
- 여전히 "뷰"를 어디서 가져올지가 문제
- 확신도(confidence)를 어떻게 정할지 주관적

---

## ML이 해결할 수 있는 것

### 1. 비선형 관계 포착

전통 팩터:
```
Return = β₁(Momentum) + β₂(Value) + β₃(Quality) + ε
```

현실은 선형이 아님:
- 모멘텀이 극단적일 때 reversal 발생
- 밸류와 퀄리티의 상호작용
- 시장 상태(bull/bear)에 따른 팩터 효과 변화

**ML 접근**:
```
Return = f(Momentum, Value, Quality, Market_State, ...)
where f is a neural network / gradient boosting / etc.
```

### 2. 동적 팩터 가중치

정적 접근 (전통):
```
Score = 0.33 * Momentum + 0.33 * Value + 0.33 * Quality
```

동적 접근 (ML):
```
Score_t = w₁(t) * Momentum + w₂(t) * Value + w₃(t) * Quality
where w(t) = neural_network(market_features_t)
```

시장 상태에 따라 어떤 팩터에 집중할지 학습.

### 3. 예측 불확실성 정량화

전통: "이 주식 수익률은 10%일 것이다"
ML + Bayesian: "수익률은 10% ± 5% (95% CI)"

**왜 중요한가?**
- 확신이 없을 때 → 포지션 축소
- 확신이 높을 때 → 포지션 확대
- 베이지안 접근과 자연스럽게 연결

---

## 업계 현황 (2024-2025)

### 주류 접근법

| 방법론 | 사용처 | 특징 |
|--------|--------|------|
| **Gradient Boosting** (XGBoost, LightGBM) | Two Sigma, Citadel | 테이블 데이터에 강함, 해석 가능 |
| **Random Forest** | 대부분의 퀀트 펀드 | 안정적, 과적합에 강함 |
| **LSTM/GRU** | 시계열 예측 | 순차적 패턴 학습 |
| **Transformer** | 최신 연구 | Attention 메커니즘, 복잡한 의존성 |
| **Gaussian Process** | 베이지안 최적화 | 불확실성 정량화 |

### 최신 트렌드 (2023-2025)

**1. Foundation Models for Finance**
- BloombergGPT, FinGPT 등 금융 특화 LLM
- 뉴스, 공시, 애널리스트 리포트 분석
- 아직 alpha 생성보다는 정보 추출에 활용

**2. Temporal Fusion Transformer (TFT)**
- Google 2021 논문
- 시계열 예측 + 해석 가능성
- 어떤 feature가 언제 중요했는지 시각화

**3. Probabilistic Forecasting**
- 점 예측 → 분포 예측
- DeepAR (Amazon), Normalizing Flows
- 리스크 관리와 직접 연결

**4. Reinforcement Learning for Portfolio**
- 포트폴리오 최적화를 RL 문제로
- 거래비용, 시장 충격 등 현실적 제약 반영
- 아직 실무 적용은 제한적

**5. Graph Neural Networks (GNN)**
- 주식 간 관계를 그래프로 모델링
- 공급망, 산업 연관성 포착
- 학계에서 활발히 연구 중

---

## 우리 프로젝트에 적용 가능한 것

### 권장: Gradient Boosting + Uncertainty

**왜?**
1. 구현 쉬움 (LightGBM, XGBoost)
2. 테이블 데이터(팩터)에 최적
3. Feature importance로 해석 가능
4. Quantile regression으로 불확실성 추정

```python
# 예시 구조
features = [momentum, value, quality, volatility, market_cap, ...]
target = next_month_return

model = LightGBMRegressor()
model.fit(features, target)

# 불확실성: Quantile regression
model_q10 = LightGBM(objective='quantile', alpha=0.1)
model_q90 = LightGBM(objective='quantile', alpha=0.9)
confidence_interval = model_q90.predict(X) - model_q10.predict(X)
```

### 도전적: LSTM/Transformer + Bayesian

**왜?**
1. 시계열 패턴 학습
2. Dropout as Bayesian Approximation
3. 논문 주제와 연결 (uncertainty quantification)

```python
# 예시 구조
sequence = [factor_scores_t-12, ..., factor_scores_t-1, factor_scores_t]
model = LSTM(dropout=0.2)  # MC Dropout for uncertainty

# 예측 시 여러 번 forward pass
predictions = [model(X, training=True) for _ in range(100)]
mean_pred = np.mean(predictions)
std_pred = np.std(predictions)  # 불확실성
```

---

## 핵심 질문에 대한 답

> "전통적 방법이 너무 rigid하고 모두가 알아서 edge가 없는 건가?"

**부분적으로 맞음:**

1. **팩터 자체는 여전히 작동**
   - 모멘텀, 밸류 프리미엄은 수십 년간 존재
   - 완전히 사라지지 않음 (행동 편향, 위험 프리미엄)

2. **하지만 단순 적용은 어려움**
   - 같은 팩터를 같은 방식으로 쓰면 crowding
   - "언제, 얼마나" 쓸지가 중요해짐

3. **ML의 진짜 가치**
   - 팩터를 대체하는 게 아니라 **더 잘 활용**
   - 비선형 관계, 시장 상태 적응, 불확실성 정량화
   - "같은 재료, 다른 요리법"

**핵심은:**
```
Traditional Factors + ML Timing + Uncertainty Quantification
= Differentiated Alpha
```

단순히 "LSTM으로 주가 예측"이 아니라,
"팩터 수익률을 예측하고, 예측의 신뢰도에 따라 포지션 조절"이 현실적.

---

## 구현 계획: A → B → C

### Step A: LightGBM + Quantile Regression (1-2일)

**목표**: 팩터 수익률 예측 + 예측 신뢰구간

```python
# 입력: 팩터 스코어 + 시장 변수
features = [momentum, value, quality, volatility, ...]

# 출력: 다음 달 수익률 분포
model_median = LightGBM(objective='quantile', alpha=0.5)
model_q10 = LightGBM(objective='quantile', alpha=0.1)
model_q90 = LightGBM(objective='quantile', alpha=0.9)

# 신뢰구간 → 포지션 사이징
confidence = 1 / (q90 - q10)  # 좁을수록 확신
position_size = base_size * confidence
```

**결과물**:
- Feature importance 시각화
- 예측 vs 실제 scatter plot
- 신뢰구간 기반 포트폴리오 백테스트

---

### Step B: LSTM + MC Dropout (3-5일)

**목표**: 시계열 패턴 학습 + Bayesian 불확실성

```python
# 입력: 과거 12개월 팩터 시퀀스
sequence = [factors_t-12, ..., factors_t-1, factors_t]

# LSTM with Dropout
model = Sequential([
    LSTM(64, dropout=0.2, return_sequences=True),
    LSTM(32, dropout=0.2),
    Dense(1)
])

# MC Dropout으로 불확실성 추정
predictions = [model(X, training=True) for _ in range(100)]
mean = np.mean(predictions)
uncertainty = np.std(predictions)
```

**결과물**:
- 시계열 패턴 학습 증명
- Epistemic vs Aleatoric uncertainty 분리
- 논문 주제와 연결: "예측의 신뢰도"

---

### Step C: Temporal Fusion Transformer (5-7일)

**목표**: 최신 아키텍처 + 해석 가능성 + 불확실성

```python
# TFT 특징:
# 1. Variable Selection: 어떤 feature가 중요한지 자동 학습
# 2. Temporal Attention: 어느 시점이 중요했는지
# 3. Quantile Output: 분포 예측 내장

from pytorch_forecasting import TemporalFusionTransformer

model = TemporalFusionTransformer(
    hidden_size=64,
    attention_head_size=4,
    output_size=7,  # 7 quantiles
)
```

**결과물**:
- Attention 시각화 (어느 시점/변수가 예측에 기여했는지)
- 내장된 불확실성 추정
- 가장 최신 트렌드 적용 증명

---

## 예상 최종 결과

```
Week 1-2: "어떤 주식을 얼마나 사야 하는가" (Factors + Optimization)
     ↓
Week 3:  "예측을 얼마나 믿을 수 있는가" (ML + Uncertainty)
     ↓
Portfolio: 확신 높으면 공격적, 낮으면 보수적
```

**면접에서의 스토리**:
> "팩터 투자의 한계를 인식하고, ML로 동적 가중치와 불확실성을 추가했습니다.
> 특히 제 thesis 주제인 'uncertainty quantification'을 투자에 적용하여,
> 예측 신뢰도에 따라 포지션을 조절하는 시스템을 구현했습니다."
