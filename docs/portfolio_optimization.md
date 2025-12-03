# 포트폴리오 최적화 (Portfolio Optimization)

## 왜 포트폴리오 최적화를 하는가?

투자의 핵심 딜레마는 **수익률과 위험의 트레이드오프**입니다.

- 높은 수익률을 원하면 → 더 큰 위험을 감수해야 함
- 위험을 줄이고 싶으면 → 기대수익률이 낮아짐

1952년 Harry Markowitz는 이 문제에 대한 혁명적 해법을 제시했습니다:

> "Don't put all your eggs in one basket" - **분산 투자**

하지만 단순히 여러 자산에 나눠 담는 것만으로는 부족합니다. **어떤 비율로** 담아야 최적인가? 이것이 포트폴리오 최적화가 답하려는 질문입니다.

---

## 핵심 개념: 위험은 줄이고 수익은 유지하기

### 분산 투자의 마법

두 자산 A, B가 있다고 합시다:
- A: 수익률 10%, 변동성 20%
- B: 수익률 10%, 변동성 20%

둘 다 같은 수익률과 위험을 가집니다. 그런데 **A와 B가 반대로 움직인다면?**

A가 떨어질 때 B가 오르고, A가 오를 때 B가 떨어진다면, 50:50으로 섞으면 **변동성이 0에 가까워집니다**. 수익률은 10%로 유지하면서요.

이것이 분산 투자의 핵심입니다: **상관관계가 낮은 자산들을 조합하면 위험은 줄이면서 수익은 유지할 수 있다.**

---

## Mean-Variance 최적화 (Markowitz, 1952)

### 문제 정의

"주어진 목표 수익률에서 위험(분산)을 최소화하는 포트폴리오는?"

수학적으로:

```
minimize    w'Σw           (포트폴리오 분산)
subject to  w'μ = μ_target (목표 수익률 달성)
            w'1 = 1        (비중 합 = 100%)
```

여기서:
- `w`: 자산 비중 벡터 (우리가 찾고 싶은 것)
- `Σ`: 공분산 행렬 (자산들이 서로 어떻게 움직이는지)
- `μ`: 기대수익률 벡터

### 해법: Lagrangian

제약이 있는 최적화 문제는 Lagrangian으로 풀 수 있습니다:

```
L = w'Σw - λ(w'μ - μ_target) - γ(w'1 - 1)
```

1차 조건 ∂L/∂w = 0에서:

```
2Σw = λμ + γ1
w = Σ^(-1)(λμ + γ1) / 2
```

### 주요 포트폴리오

#### 1. Global Minimum Variance (GMV) 포트폴리오

**목표 수익률 제약 없이** 위험만 최소화:

```
w_gmv = Σ^(-1) * 1 / (1' * Σ^(-1) * 1)
```

**의미**: 가장 보수적인 투자자가 선택하는 포트폴리오

#### 2. Maximum Sharpe Ratio (Tangency) 포트폴리오

**샤프 비율**(단위 위험당 초과수익)을 최대화:

```
Sharpe = (w'μ - r_f) / √(w'Σw)

w_tan = Σ^(-1) * (μ - r_f) / (1' * Σ^(-1) * (μ - r_f))
```

**의미**: 가장 효율적인 포트폴리오. 무위험자산과 조합하면 모든 투자자에게 최적.

### 효율적 프론티어 (Efficient Frontier)

다양한 목표 수익률에서 최소 분산 포트폴리오를 구하면 **효율적 프론티어** 곡선이 그려집니다.

```
수익률
  ^
  |         * Tangency
  |       /
  |     / ← 효율적 프론티어
  |   /
  | * GMV
  +-------------------> 변동성
```

- 곡선 위의 점: 효율적 (같은 위험에서 최대 수익)
- 곡선 아래의 점: 비효율적 (개선 가능)

---

## Mean-Variance의 한계

Markowitz 모델은 혁명적이지만 실무에서 몇 가지 문제가 있습니다:

### 1. 추정 오차에 민감

- 기대수익률 `μ`를 추정하기가 매우 어려움
- 작은 추정 오차 → 극단적인 비중 변화
- "Garbage in, garbage out"

### 2. 극단적 비중

- 최적화 결과가 특정 자산에 몰리는 경향
- 예: "A에 200% 롱, B에 -100% 숏" (비현실적)

### 3. 시장 정보를 무시

- 시장이 이미 많은 정보를 가격에 반영하고 있음
- 순수하게 역사적 데이터만 사용하는 것은 비효율적

---

## Black-Litterman 모델 (1992)

Black-Litterman은 위 문제들을 해결하기 위해 **베이지안 접근법**을 제안했습니다.

### 핵심 아이디어

```
시장 균형 수익률 (Prior) + 투자자 뷰 (Likelihood) = 최적 기대수익률 (Posterior)
```

1. **시장이 효율적이라고 가정**하고 현재 시장 비중에서 "implied" 기대수익률 계산
2. **투자자의 뷰**를 확신도와 함께 입력
3. **베이지안 업데이트**로 둘을 결합

### Step 1: Prior - 균형 수익률

시장이 균형 상태라면, 현재 시장 비중은 "최적"입니다.
역으로, 현재 비중에서 implied 되는 기대수익률을 계산할 수 있습니다:

```
π = δ × Σ × w_mkt

π: 균형 기대수익률
δ: 위험회피계수 (보통 2.5)
Σ: 공분산 행렬
w_mkt: 시장 비중 (시가총액 가중)
```

**베이지안 해석**: π가 "사전 분포"의 평균

### Step 2: 투자자 뷰

투자자는 자신만의 예측(뷰)을 가질 수 있습니다:

**절대 뷰**: "삼성전자 수익률이 15%일 것이다"
```
P × μ = Q + ε

P = [1, 0, 0, ...]  (삼성전자만 1)
Q = 0.15
```

**상대 뷰**: "삼성전자가 SK하이닉스보다 5% 아웃퍼폼할 것이다"
```
P = [1, -1, 0, ...]  (삼성 +1, SK -1)
Q = 0.05
```

뷰의 **불확실성** Ω도 함께 지정합니다. 확신이 높을수록 Ω가 작아집니다.

### Step 3: Posterior - 베이지안 업데이트

Prior와 Likelihood를 결합하여 Posterior 계산:

```
μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)π + P'Ω^(-1)Q]

τ: 스케일링 파라미터 (보통 0.05)
```

**결과 해석**:
- 뷰가 없으면 → μ_BL ≈ π (시장 균형)
- 확신도가 높은 뷰 → μ_BL이 뷰 방향으로 크게 이동
- 확신도가 낮은 뷰 → μ_BL이 약간만 이동

---

## 베이지안 프레임워크로 이해하기

Black-Litterman은 베이즈 정리의 직접적 응용입니다:

```
P(μ|Q) ∝ P(Q|μ) × P(μ)

Prior:      μ ~ N(π, τΣ)
Likelihood: Q|μ ~ N(Pμ, Ω)
Posterior:  μ|Q ~ N(μ_BL, Σ_BL)
```

- **Prior**: 시장 균형에 대한 "사전 믿음"
- **Likelihood**: 뷰가 주어졌을 때 μ의 분포
- **Posterior**: 뷰를 반영한 "사후 믿음"

이 프레임워크의 장점:
1. 뷰가 없으면 시장 균형으로 자연스럽게 회귀
2. 확신도에 따라 뷰의 영향력이 조절됨
3. 여러 뷰를 일관되게 통합할 수 있음

---

## 구현: quant-core (Rust)

성능을 위해 핵심 계산은 Rust로 구현하고, Python에서 호출합니다.

### 사용법

```python
import numpy as np
from quant_core import MeanVarianceOptimizer, BlackLitterman, sample_covariance

# 수익률 데이터에서 공분산 추정
returns = ...  # (T x n) numpy array
cov = sample_covariance(returns) * 252  # 연율화

# Mean-Variance 최적화
expected_returns = np.array([0.10, 0.15, 0.12])
mvo = MeanVarianceOptimizer(cov, expected_returns)

# 최소 분산 포트폴리오
gmv_weights = mvo.min_variance()

# 최대 샤프 포트폴리오
tangency_weights = mvo.max_sharpe(risk_free_rate=0.03)

# Black-Litterman
market_weights = np.array([0.5, 0.3, 0.2])
bl = BlackLitterman(cov, market_weights, risk_aversion=2.5, tau=0.05)

# 균형 수익률 확인
equilibrium = bl.equilibrium_returns()

# 뷰 추가
bl.add_absolute_view(asset_index=0, view_return=0.20, confidence=0.8)
bl.add_relative_view(long_index=1, short_index=2, relative_return=0.05, confidence=0.6)

# Posterior 수익률과 최적 비중
posterior = bl.posterior_returns()
optimal_weights = bl.optimal_weights()
```

### API 참조

#### MeanVarianceOptimizer

| 메서드 | 설명 |
|--------|------|
| `min_variance()` | GMV 포트폴리오 비중 |
| `max_sharpe(rf)` | Tangency 포트폴리오 비중 |
| `target_return(μ)` | 목표 수익률 포트폴리오 |
| `efficient_frontier(n)` | 효율적 프론티어 계산 |
| `portfolio_stats(w, rf)` | (수익률, 변동성, 샤프) |

#### BlackLitterman

| 메서드 | 설명 |
|--------|------|
| `equilibrium_returns()` | Prior (시장 균형 수익률) |
| `add_absolute_view()` | 절대 뷰 추가 |
| `add_relative_view()` | 상대 뷰 추가 |
| `posterior_returns()` | Posterior 기대수익률 |
| `optimal_weights()` | BL 최적 비중 |
| `clear_views()` | 뷰 초기화 |

---

## 개념 점검 Q&A

### Q1. Mean-Variance 최적화에서 공분산 행렬의 역할은?

공분산 행렬은 **자산 간의 상관관계**를 담고 있습니다. 이것이 있어야 분산 투자의 효과를 계산할 수 있습니다.

- 상관관계가 낮은 자산들 → 분산 투자 효과 큼
- 상관관계가 높은 자산들 → 분산 투자 효과 작음

### Q2. 왜 Black-Litterman이 필요한가?

Mean-Variance의 주요 문제점:
1. **기대수익률 추정의 어려움** - BL은 시장 균형에서 시작
2. **극단적 비중** - BL은 시장 비중에서 점진적으로 이동
3. **시장 정보 무시** - BL은 시장이 효율적이라고 가정

### Q3. Black-Litterman의 τ (tau)는 무엇인가?

τ는 **Prior의 불확실성**을 조절하는 파라미터입니다.

- τ가 작으면 → Prior(시장 균형)에 대한 확신이 높음 → 뷰의 영향 작음
- τ가 크면 → Prior에 대한 확신이 낮음 → 뷰의 영향 큼

실무에서는 보통 0.01 ~ 0.10 사이 값을 사용합니다.

### Q4. 효율적 프론티어 위의 모든 점이 "좋은" 포트폴리오인가?

네, 모두 **Pareto 최적**입니다. 같은 위험에서 더 높은 수익을 얻거나, 같은 수익에서 더 낮은 위험을 가진 포트폴리오가 존재하지 않습니다.

투자자의 위험 선호도에 따라 프론티어 위의 다른 점을 선택합니다:
- 보수적 투자자 → GMV 근처
- 공격적 투자자 → 고수익/고위험 영역

### Q5. Black-Litterman에서 확신도(confidence)는 어떻게 설정하는가?

확신도는 0과 1 사이 값으로:
- **0.5**: "그냥 그런 것 같아"
- **0.8**: "꽤 확신함"
- **0.95**: "거의 확실함"

실무에서는:
- 정량적 분석 기반 → 높은 확신도
- 직관/정성적 판단 → 낮은 확신도
- 매크로 이벤트 전망 → 중간 확신도

---

## 더 알아보기

- Markowitz, H. (1952). "Portfolio Selection". *The Journal of Finance*.
- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization". *Financial Analysts Journal*.
- Idzorek, T. (2007). "A Step-by-Step Guide to the Black-Litterman Model".
