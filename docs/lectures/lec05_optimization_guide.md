# Lecture 05: 최적화 방법 (Optimization Methods) 학습 가이드

**강의 자료**: `lec05_optimization.pdf`
**과목**: 금융수치해석기법 (Numerical Methods in Finance)
**대상**: KAIST MFE

---

## 직관적 오버뷰: 왜 최적화인가?

### 금융의 근본 질문

금융공학의 거의 모든 문제는 결국 **"최선의 선택은 무엇인가?"**로 귀결됩니다.

```
포트폴리오 매니저: "리스크 대비 수익을 최대화하는 자산 배분은?"
트레이더:         "이 옵션의 공정 가격은?" (= 복제 비용 최소화)
리스크 매니저:    "99% 확률로 손실이 이 값 이하가 되는 VaR은?"
퀀트:            "시장 가격에 맞는 모델 파라미터는?" (= 오차 최소화)
```

이 모든 질문의 수학적 표현:

$$x^* = \arg\min_x f(x) \quad \text{또는} \quad \arg\max_x f(x)$$

### 문제의식: 왜 수치적 방법이 필요한가?

**이상적 세계**: 미분해서 0이 되는 점을 해석적으로 구한다.

$$f'(x) = 0 \quad \Rightarrow \quad x^* = \text{closed-form solution}$$

**현실 세계**: 대부분의 경우 해석해가 없다.

| 문제 | 왜 해석해가 없는가? |
|-----|-------------------|
| Black-Scholes 내재변동성 | $\sigma$에 대해 풀 수 없는 초월방정식 |
| 포트폴리오 최적화 (제약조건) | 부등식 제약이 있으면 KKT 조건이 복잡 |
| 수익률 곡선 피팅 | 비선형 모델 (Nelson-Siegel 등) |
| 옵션 모델 캘리브레이션 | 다차원 비선형 최소자승 |

**따라서**: 컴퓨터를 이용한 **반복적 근사(iterative approximation)**가 필수

### 최적화의 사고 체계 (House of Thoughts)

모든 수치 최적화 방법은 다음 프레임워크 안에서 이해할 수 있습니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    최적화의 핵심 질문                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 어디로 갈 것인가? (Search Direction)                         │
│      └─ "현재 위치에서 어느 방향으로 가면 f(x)가 줄어드는가?"          │
│                                                                 │
│   2. 얼마나 갈 것인가? (Step Size)                                │
│      └─ "그 방향으로 얼마나 이동해야 하는가?"                        │
│                                                                 │
│   3. 언제 멈출 것인가? (Stopping Criterion)                       │
│      └─ "충분히 좋은 해에 도달했는가?"                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 방법들의 계보: 정보량에 따른 분류

사용하는 **정보의 양**에 따라 방법이 달라집니다:

```
정보 많음                                                    정보 적음
    │                                                           │
    ▼                                                           ▼
┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐
│Newton  │     │Quasi-  │     │Steepest│     │Golden  │     │Nelder- │
│Method  │────▶│Newton  │────▶│Descent │────▶│Section │────▶│Mead    │
│        │     │(BFGS)  │     │        │     │        │     │        │
└────────┘     └────────┘     └────────┘     └────────┘     └────────┘
    │              │              │              │              │
    ▼              ▼              ▼              ▼              ▼
  f, f', f''    f, f'          f, f'           f만            f만
  (Hessian)    (Hessian 근사)   (Gradient)      사용           사용

  수렴: 빠름 ◀──────────────────────────────────────────▶ 수렴: 느림
  비용: 높음 ◀──────────────────────────────────────────▶ 비용: 낮음
```

### 핵심 트레이드오프

**"공짜 점심은 없다"** - 모든 방법은 트레이드오프가 있습니다:

| 트레이드오프 | 설명 |
|------------|------|
| **속도 vs 안정성** | Newton은 빠르지만 발산할 수 있음. Steepest descent는 느리지만 안정적 |
| **정확도 vs 비용** | Hessian을 정확히 계산하면 빠르지만 비용이 큼 |
| **일반성 vs 효율성** | Nelder-Mead는 어디든 적용 가능하지만 비효율적 |
| **전역 vs 국소** | 대부분의 방법은 국소 최적해만 찾음 |

### 실무자의 선택 기준

```
                    도함수 계산 가능?
                          │
              ┌───────────┴───────────┐
              │                       │
             Yes                      No
              │                       │
              ▼                       ▼
        Hessian 계산 가능?          Nelder-Mead
              │                    Golden Section
      ┌───────┴───────┐
      │               │
     Yes              No
      │               │
      ▼               ▼
   Newton          BFGS
   (가장 빠름)      (실무 표준)
```

### 이 강의에서 배울 것

```
Part 1: 기초 - 해찾기 (Root Finding)
        └─ 최적화의 본질은 f'(x) = 0 풀기

Part 2: 1차원 최적화
        └─ Newton, Golden Section

Part 3: 다차원 최적화
        └─ Steepest Descent → Newton → BFGS

Part 4: 특수 문제
        └─ 직접탐색법 (Nelder-Mead)
        └─ 비선형 최소자승 (Gauss-Newton, Levenberg-Marquardt)
        └─ 비선형 연립방정식 (Newton-Raphson, Broyden)
```

---

## 목차

1. [비선형방정식의 해찾기 (Root Finding)](#1-비선형방정식의-해찾기)
2. [최적화 문제 개요](#2-최적화-문제-개요)
3. [1차원 최적화](#3-1차원-최적화)
4. [다차원 최적화](#4-다차원-최적화)
5. [직접탐색법 (Direct Search)](#5-직접탐색법)
6. [비선형 최소자승법](#6-비선형-최소자승법)
7. [비선형 연립방정식](#7-비선형-연립방정식)
8. [방법 선택 가이드](#8-방법-선택-가이드)

---

## 1. 비선형방정식의 해찾기

### 왜 중요한가?

최적화 문제의 핵심은 $f'(x) = 0$인 점을 찾는 것입니다. 즉, **해찾기(root finding)가 최적화의 기초**입니다.

### 1.1 Bisection Method (이분법)

**핵심 아이디어**: 구간을 반으로 나누며 해를 좁혀감

```
조건: f(a)와 f(b)의 부호가 다름 (중간값 정리)

반복:
1. c = (a + b) / 2
2. if sign(f(a)) ≠ sign(f(c)): b = c
   else: a = c
3. |b - a| < η 될 때까지 반복
```

**장점**: 항상 수렴, 구현 간단
**단점**: 수렴 속도 느림 (선형 수렴)

### 1.2 Fixed Point Method (고정점 반복법)

**핵심 아이디어**: $f(x) = 0$을 $x = g(x)$ 형태로 변환

$$x_{new} = g(x_{old})$$

**예시**: $f(x) = x - x^{7/5} + 1/5 = 0$ 풀기

두 가지 iteration function:
- $g_1(x) = x^{7/5} - 1/5$ → 발산!
- $g_2(x) = (x + 1/5)^{5/7}$ → 수렴!

**수렴 조건**: $|g'(x^*)| < 1$ (해 근처에서 기울기가 1보다 작아야 함)

### 1.3 Newton's Method (뉴턴 방법)

**핵심 아이디어**: 접선이 x축과 만나는 점으로 이동

$$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$$

**유도**:
```
Taylor 전개: f(x + Δx) ≈ f(x) + f'(x)Δx

f(x + Δx) = 0이 되는 Δx 찾기:
Δx = -f(x) / f'(x)
```

**장점**: 이차 수렴 (quadratic convergence) - 매우 빠름
**단점**: 도함수 필요, 초기값에 민감

---

## 2. 최적화 문제 개요

### 일반적인 최적화 문제

$$x^* = \arg\min_{x \in \mathbb{R}^n} f(x)$$

- $f$: 목적함수 (objective function)
- $x^*$: 최적해 (solution)

### 최적화 문제의 종류

| 유형 | 설명 | 예시 |
|-----|------|------|
| **Linear Programming (LP)** | $f(x)$와 제약조건이 선형 | 포트폴리오 비용 최소화 |
| **Quadratic Programming (QP)** | $f(x)$가 이차, 제약조건 선형 | Markowitz 포트폴리오 |
| **Convex Programming** | $f(x)$와 제약집합이 볼록 | 리스크 패리티 |
| **Nonlinear Least Squares** | $f(x) = \frac{1}{2}\sum r_i(x)^2$ | 옵션 캘리브레이션 |

### Gradient와 Hessian

**Gradient (기울기 벡터)**:
$$\nabla f(x) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

**Hessian (곡률 행렬)**:
$$\nabla^2 f(x) = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}$$

### 최솟값 조건

| 조건 | 수식 | 의미 |
|-----|------|------|
| **FOC (1차 조건)** | $\nabla f(x^*) = 0$ | 기울기가 0 |
| **SOC (2차 조건)** | $z'\nabla^2 f(x^*)z \geq 0$ | Hessian이 양정치 |

FOC는 만족하나 SOC를 만족하지 못하면 → **안장점 (saddle point)**

---

## 3. 1차원 최적화

### 3.1 Newton's Method for Optimization

**핵심**: $f'(x) = 0$을 풀기 위해 Newton 방법 적용

$$x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}$$

**직관**:
```
이동량 = 기울기 / 곡률
       = "얼마나 경사진가" / "얼마나 휘어있나"
```

**예시**: $f(x) = x^2$ 최소화
- $f'(x) = 2x$, $f''(x) = 2$
- $x_{k+1} = x_k - \frac{2x_k}{2} = 0$ → **한 번에 수렴!**

### 3.2 Golden Section Search (황금분할 탐색)

**용도**: 도함수 없이 unimodal 함수의 최솟값 찾기

**핵심**: 구간을 황금비 $\tau = \frac{\sqrt{5}-1}{2} \approx 0.618$로 줄여감

```
구간 [a, b]에서:
1. x₁ = a + (1-τ)(b-a)
2. x₂ = a + τ(b-a)
3. f(x₁) < f(x₂)이면 b = x₂
   아니면 a = x₁
4. 반복
```

**장점**: 도함수 불필요
**단점**: 선형 수렴, unimodal 함수에만 적용

---

## 4. 다차원 최적화

### 4.1 Steepest Descent (최급강하법)

**핵심**: 가장 가파른 내리막 방향 (음의 gradient)으로 이동

$$x_{k+1} = x_k - \alpha \nabla f(x_k)$$

여기서 $\alpha$는 **line search**로 결정:
$$\alpha^* = \arg\min_\alpha f(x_k - \alpha \nabla f(x_k))$$

**장점**: 구현 간단, 항상 감소 방향
**단점**: 수렴 느림 (지그재그 현상), ill-conditioned 문제에 취약

```
Steepest descent의 문제점:

  ╭─────────────────────────╮
  │  ↗↙↗↙↗↙↗↙↗↙↗●          │  ← 지그재그로 수렴
  │    (등고선이 길쭉한 경우)    │
  ╰─────────────────────────╯
```

### 4.2 Newton's Method (다차원)

**핵심**: 2차 근사의 최솟값으로 바로 이동

$$\nabla^2 f(x_k) \cdot s_k = -\nabla f(x_k)$$
$$x_{k+1} = x_k + s_k$$

**단변량 vs 다변량**:

| 단변량 | 다변량 |
|-------|-------|
| $f'(x)$ | $\nabla f(x)$ |
| $f''(x)$ | $\nabla^2 f(x)$ (Hessian) |
| 나눗셈 | 역행렬 (또는 연립방정식) |

**장점**: 이차 수렴 (매우 빠름)
**단점**: Hessian 계산 비용, Hessian이 양정치가 아니면 문제

### 4.3 Quasi-Newton (BFGS)

**핵심**: Hessian을 직접 계산하지 않고 근사

$$B_{k+1} = B_k + U$$

여기서 $U$는 BFGS 업데이트:
$$U = \frac{y^{(k)} {y^{(k)}}'}{{y^{(k)}}' s^{(k)}} - \frac{(B_k s^{(k)})(B_k s^{(k)})'}{s^{(k)'} B_k s^{(k)}}$$

**장점**: Hessian 계산 불필요, superlinear 수렴
**단점**: Newton보다는 느림

---

## 5. 직접탐색법 (Direct Search)

### 언제 사용하나?

- 도함수가 존재하지 않거나 구할 수 없는 경우
- 함수 계산이 매우 비싼 경우 (예: 시뮬레이션)
- 함수값에 잡음이 포함된 경우 (예: 몬테카를로)
- 완벽한 최적값보다 개선에만 관심있는 경우

### 5.1 Nelder-Mead Simplex

**핵심**: n차원에서 (n+1)개 꼭짓점으로 이루어진 simplex를 변형하며 탐색

**연산들**:

| 연산 | 설명 | 조건 |
|-----|------|------|
| **Reflection** | 최악의 점을 반대로 반사 | $x_R = (1+\rho)\bar{x} - \rho x_{n+1}$ |
| **Expansion** | 좋으면 더 확장 | $f(x_R) < f(x_1)$ |
| **Contraction** | 안 좋으면 수축 | $f(x_R) > f(x_n)$ |
| **Shrink** | 모두 안 좋으면 전체 축소 | 모든 조건 실패 |

**전형적 파라미터**: $\rho = 1$, $\psi = 0.5$, $\sigma = 2$

```
Nelder-Mead 시각화 (2D):

    Worst
      ●
     / \
    /   \       → Reflect →      ●───●
   /     \                        \   \
  ●───────●                        ●───● New
 Best                                    Best
```

**장점**: 도함수 불필요, 잡음에 강건
**단점**: 고차원에서 비효율적, 수렴 보장 없음

---

## 6. 비선형 최소자승법 (Nonlinear Least Squares)

### 문제 정의

$$g(x) = \frac{1}{2} r(x)' r(x) = \frac{1}{2} \sum_{i=1}^m r_i(x)^2$$

여기서 $r_i(x) = y_i - f(t_i, x)$ (잔차, residual)

**금융 응용**: 옵션 변동성 캘리브레이션, 수익률 곡선 피팅

### Gradient와 Hessian

$$\nabla g(x) = \nabla r(x)' r(x)$$

$$\nabla^2 g(x) = \nabla r(x)' \nabla r(x) + S(x)$$

여기서 $\nabla r(x)$는 **Jacobian**, $S(x)$는 2차 미분 항

### 6.1 Gauss-Newton Method

**핵심**: $S(x) \approx 0$으로 근사 (잔차가 작다고 가정)

$$(\nabla r(x_k)' \nabla r(x_k)) s_{GN}^k = -\nabla r(x_k)' r(x_k)$$

**장점**: 2차 미분 계산 불필요
**단점**: 잔차가 크면 수렴 느림

### 6.2 Levenberg-Marquardt Method

**핵심**: Gauss-Newton + 정규화

$$(\nabla r(x_k)' \nabla r(x_k) + \mu_k I) s_{LM}^k = -\nabla r(x_k)' r(x_k)$$

- $\mu_k$ 크면: Steepest descent처럼 동작 (안정적)
- $\mu_k$ 작으면: Gauss-Newton처럼 동작 (빠름)

**장점**: Gauss-Newton보다 안정적, 널리 사용됨
**단점**: $\mu_k$ 조절 필요

---

## 7. 비선형 연립방정식

### 문제 정의

$$F(y) = 0, \quad F(y) = \begin{pmatrix} f_1(y) \\ \vdots \\ f_n(y) \end{pmatrix}$$

### Jacobian Matrix

$$\nabla F(y) = \begin{pmatrix} \frac{\partial f_1}{\partial y_1} & \cdots & \frac{\partial f_1}{\partial y_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_n}{\partial y_1} & \cdots & \frac{\partial f_n}{\partial y_n} \end{pmatrix}$$

### 7.1 반복법 (Jacobi, Gauss-Seidel, SOR)

**Jacobi**: 모든 변수를 이전 값으로 업데이트
$$y_i^{k+1} = g_i(y_1^k, \ldots, y_{i-1}^k, y_{i+1}^k, \ldots, y_n^k)$$

**Gauss-Seidel**: 이미 계산된 새 값 바로 사용
$$y_i^{k+1} = g_i(y_1^{k+1}, \ldots, y_{i-1}^{k+1}, y_{i+1}^k, \ldots, y_n^k)$$

**수렴 기준**: $\frac{|y_i^{k+1} - y_i^k|}{|y_i^k| + 1} < \epsilon$

### 7.2 Newton-Raphson Method

**핵심**: 선형 근사 후 연립방정식 풀기

$$F(y) \approx F(y^k) + \nabla F(y^k)(y - y^k) = 0$$

$$y^{k+1} = y^k - [\nabla F(y^k)]^{-1} F(y^k)$$

**실제 구현**: 역행렬 대신 연립방정식 $\nabla F(y^k) \cdot s = -F(y^k)$ 풀기

### 7.3 Broyden's Method

**핵심**: Jacobian을 직접 계산하지 않고 업데이트

$$B^{(k+1)} = B^{(k)} + \frac{(dF^{(k)} - B^{(k)} s^{(k)}) {s^{(k)}}'}{s^{(k)'} s^{(k)}}$$

여기서 $dF^{(k)} = F(y^{k+1}) - F(y^k)$

**장점**: Jacobian 계산 불필요
**단점**: Newton보다 느림

---

## 8. 방법 선택 가이드

### 문제 유형별 추천

| 문제 | 추천 방법 |
|-----|----------|
| 도함수 있음, 빠른 수렴 필요 | Newton's method |
| 도함수 있음, Hessian 비쌈 | BFGS (Quasi-Newton) |
| 도함수 없음 | Nelder-Mead, Golden section |
| 최소자승 문제 | Levenberg-Marquardt |
| 비선형 연립방정식 | Newton-Raphson, Broyden |

### 수렴 속도 비교

| 방법 | 수렴 차수 | 한 번 반복 비용 |
|-----|----------|---------------|
| Bisection | 선형 | 낮음 |
| Steepest Descent | 선형 | 중간 |
| BFGS | 초선형 | 중간 |
| Newton | 이차 | 높음 |
| Gauss-Newton | 이차 (잔차 작을 때) | 중간 |

### 금융 응용 예시

| 응용 | 추천 방법 |
|-----|----------|
| **옵션 내재변동성** | Newton-Raphson |
| **포트폴리오 최적화** | BFGS, 내점법 |
| **모델 캘리브레이션** | Levenberg-Marquardt |
| **VaR 계산** | Bisection |
| **수익률곡선 피팅** | Gauss-Newton |

---

## Python 구현 참조

이 저장소의 관련 파일:
- `src/numerical/linear_system_iterative.py` - Jacobi, Gauss-Seidel, SOR
- `docs/course/problems/problem_04_newton_method.md` - Newton 방법 상세 해설
- `docs/course/problems/problem_05_sor_method.md` - SOR 방법

### scipy 활용

```python
from scipy.optimize import minimize, fsolve, least_squares

# 다차원 최적화
result = minimize(f, x0, method='BFGS')       # Quasi-Newton
result = minimize(f, x0, method='Nelder-Mead') # Direct search

# 비선형 연립방정식
root = fsolve(F, y0)

# 비선형 최소자승
result = least_squares(residuals, x0, method='lm')  # Levenberg-Marquardt
```

---

## 핵심 요약

```
최적화 = f'(x) = 0 풀기

1차원:
  - Newton: x_{k+1} = x_k - f'(x)/f''(x)  [빠름, 도함수 필요]
  - Golden section: 구간 축소  [느림, 도함수 불필요]

다차원:
  - Steepest descent: -∇f 방향으로 이동  [느림, 안정적]
  - Newton: Hessian 역행렬 사용  [빠름, 비쌈]
  - BFGS: Hessian 근사  [빠름, 실용적]
  - Nelder-Mead: 함수값만 사용  [도함수 불필요]

최소자승:
  - Gauss-Newton: Hessian ≈ J'J
  - Levenberg-Marquardt: 정규화 추가

비선형 시스템:
  - Newton-Raphson: Jacobian 사용
  - Broyden: Jacobian 근사
```

---

**작성일**: 2025-12-11
**참조**: KAIST MFE 금융수치해석기법 Lecture 05
