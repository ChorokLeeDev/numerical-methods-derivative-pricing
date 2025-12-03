# 문제 5: SOR 방법 (15점)

## 📚 강의 노트 참조
- **Lecture 02: Linear System of Equations**
  - p.18-22: Iterative methods (Jacobi, Gauss-Seidel, SOR)
  - p.19: Matrix splitting and iteration matrices
  - p.21: SOR convergence analysis

---

## 📋 원본 문제

> **문제 5 (15점)**
>
> 연립 방정식 $A\mathbf{x} = \mathbf{b}$에서:
>
> $$A = \begin{pmatrix} 2 & 4 \\ 2 & -4 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 6 \\ -2 \end{pmatrix}, \quad \omega = 2$$
>
> SOR 방법에서 $M\mathbf{x}^{(k+1)} = N\mathbf{x}^{(k)} + \omega\mathbf{b}$로 표현할 때:
>
> **(1)** $M$과 $N$을 구하시오. (10점)
>
> **(2)** $\mathbf{e}^{(k)} = (-1, 1)^T$일 때 $\mathbf{e}^{(k+1)}$을 구하시오. (5점)
>
> 단, $\mathbf{e}^{(k)} = \mathbf{x}^{(k)} - \mathbf{x}^*$는 오차 벡터이다.

---

## 🎯 먼저 기본 개념부터!

### 연립방정식이란?

여러 개의 방정식을 **동시에** 만족하는 해를 찾는 것:

$$\begin{cases} 2x + 4y = 6 \\ 2x - 4y = -2 \end{cases}$$

행렬로 쓰면: $A\mathbf{x} = \mathbf{b}$

### 연립방정식을 푸는 두 가지 방법

| 방법 | 특징 | 예시 |
|-----|------|-----|
| **직접법** | 정확한 답을 한 번에 | 가우스 소거법 |
| **반복법** | 추측 → 개선 → 추측 → ... | SOR |

**반복법의 장점**: 큰 행렬에서 효율적, 메모리 절약

---

## 💡 SOR이 뭔가요?

### SOR = Successive Over-Relaxation

한국어로: **연속적 과완화법**

### 기본 아이디어

1. 답을 **추측**합니다
2. 추측을 **개선**합니다
3. 충분히 정확해질 때까지 **반복**합니다

### "Over-Relaxation"이 뭔가요?

**비유로 설명**:

```
목표: 문 손잡이까지 걸어가기

일반적 방법 (Gauss-Seidel):
  한 발 → 한 발 → 한 발 → 도착

Over-Relaxation (SOR):
  "에이, 좀 더 가자!" → 성큼성큼 → 더 빨리 도착!

ω(오메가) = "얼마나 과하게 갈지" 조절
  ω = 1: 정상 보폭
  ω > 1: 성큼성큼 (빨리 수렴)
  ω < 1: 조심조심 (더 안정적)
```

---

## 💡 SOR 공식 유도

### Step 1: 행렬 $A$를 세 부분으로 분해

어떤 행렬이든 이렇게 나눌 수 있습니다:

$$A = D + L + U$$

- **D**: 대각선 부분만 (Diagonal)
- **L**: 대각선 아래 부분 (Lower)
- **U**: 대각선 위 부분 (Upper)

우리 문제에서:

$$A = \begin{pmatrix} 2 & 4 \\ 2 & -4 \end{pmatrix}$$

분해하면:

```
     A          =      D       +       L       +       U
┌───┬───┐         ┌───┬───┐     ┌───┬───┐     ┌───┬───┐
│ 2 │ 4 │    =    │ 2 │ 0 │  +  │ 0 │ 0 │  +  │ 0 │ 4 │
├───┼───┤         ├───┼───┤     ├───┼───┤     ├───┼───┤
│ 2 │-4 │         │ 0 │-4 │     │ 2 │ 0 │     │ 0 │ 0 │
└───┴───┘         └───┴───┘     └───┴───┘     └───┴───┘
```

### Step 2: SOR의 반복 공식

SOR 방법의 핵심 공식:

$$M\mathbf{x}^{(k+1)} = N\mathbf{x}^{(k)} + \omega\mathbf{b}$$

여기서:
- $M = D + \omega L$
- $N = (1-\omega)D - \omega U$

---

## 💡 (1) M과 N 구하기

### $\omega = 2$ 대입

**M 계산**:

$$M = D + \omega L = D + 2L$$

$$M = \begin{pmatrix} 2 & 0 \\ 0 & -4 \end{pmatrix} + 2 \times \begin{pmatrix} 0 & 0 \\ 2 & 0 \end{pmatrix}$$

$$M = \begin{pmatrix} 2 & 0 \\ 0 & -4 \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 4 & 0 \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 4 & -4 \end{pmatrix}$$

**N 계산**:

$$N = (1-\omega)D - \omega U = (1-2)D - 2U = -D - 2U$$

$$N = -\begin{pmatrix} 2 & 0 \\ 0 & -4 \end{pmatrix} - 2 \times \begin{pmatrix} 0 & 4 \\ 0 & 0 \end{pmatrix}$$

$$N = \begin{pmatrix} -2 & 0 \\ 0 & 4 \end{pmatrix} + \begin{pmatrix} 0 & -8 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} -2 & -8 \\ 0 & 4 \end{pmatrix}$$

### ✅ (1) 답

$$\boxed{M = \begin{pmatrix} 2 & 0 \\ 4 & -4 \end{pmatrix}, \quad N = \begin{pmatrix} -2 & -8 \\ 0 & 4 \end{pmatrix}}$$

---

## 💡 (2) 오차 벡터 $\mathbf{e}^{(k+1)}$ 계산

### 오차 벡터란?

$$\mathbf{e}^{(k)} = \mathbf{x}^{(k)} - \mathbf{x}^*$$

**현재 추측값**과 **정답** 사이의 차이입니다.

### 오차가 어떻게 전파되나요?

원래 반복식:
$$M\mathbf{x}^{(k+1)} = N\mathbf{x}^{(k)} + \omega\mathbf{b}$$

정답 $\mathbf{x}^*$도 이 식을 만족합니다 (왜냐면 정답이니까):
$$M\mathbf{x}^* = N\mathbf{x}^* + \omega\mathbf{b}$$

두 식을 빼면:
$$M(\mathbf{x}^{(k+1)} - \mathbf{x}^*) = N(\mathbf{x}^{(k)} - \mathbf{x}^*)$$

$$M\mathbf{e}^{(k+1)} = N\mathbf{e}^{(k)}$$

**핵심**: 오차의 전파는 $\mathbf{b}$와 무관하고, 오직 $M$과 $N$에만 의존!

---

### 계산해 봅시다

주어진 조건: $\mathbf{e}^{(k)} = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$

**Step 1**: $N\mathbf{e}^{(k)}$ 계산

$$N\mathbf{e}^{(k)} = \begin{pmatrix} -2 & -8 \\ 0 & 4 \end{pmatrix} \begin{pmatrix} -1 \\ 1 \end{pmatrix}$$

첫 번째 행: $(-2) \times (-1) + (-8) \times 1 = 2 - 8 = -6$

두 번째 행: $0 \times (-1) + 4 \times 1 = 0 + 4 = 4$

$$N\mathbf{e}^{(k)} = \begin{pmatrix} -6 \\ 4 \end{pmatrix}$$

**Step 2**: $M\mathbf{e}^{(k+1)} = \begin{pmatrix} -6 \\ 4 \end{pmatrix}$ 풀기

$$\begin{pmatrix} 2 & 0 \\ 4 & -4 \end{pmatrix} \begin{pmatrix} e_1 \\ e_2 \end{pmatrix} = \begin{pmatrix} -6 \\ 4 \end{pmatrix}$$

첫 번째 방정식: $2e_1 = -6$ → $e_1 = -3$

두 번째 방정식: $4e_1 - 4e_2 = 4$
$$4(-3) - 4e_2 = 4$$
$$-12 - 4e_2 = 4$$
$$-4e_2 = 16$$
$$e_2 = -4$$

### ✅ (2) 답

$$\boxed{\mathbf{e}^{(k+1)} = \begin{pmatrix} -3 \\ -4 \end{pmatrix}}$$

---

### 이게 무슨 의미인가요?

오차의 크기를 비교해 봅시다:

$$\|\mathbf{e}^{(k)}\| = \sqrt{(-1)^2 + 1^2} = \sqrt{2} \approx 1.41$$

$$\|\mathbf{e}^{(k+1)}\| = \sqrt{(-3)^2 + (-4)^2} = \sqrt{9+16} = 5$$

**오차가 커졌습니다!** 😱

이 SOR은 $\omega = 2$로 **발산**하고 있습니다.

### 왜 발산하나요?

일반적으로 SOR이 수렴하려면:

$$0 < \omega < 2$$

$\omega = 2$는 **경계값**이라서 이 특정 행렬에서는 수렴하지 않습니다.

```
ω 값에 따른 수렴 속도:

수렴 속도
    ↑
    │      ╱╲
    │     ╱  ╲
    │    ╱    ╲
    │   ╱      ╲
    │  ╱        ╲
    └──────────────→ ω
    0    1    2
         ↑
      최적 ω (문제마다 다름)
```

---

## ✅ 최종 답안

### (1) M과 N

$$M = \begin{pmatrix} 2 & 0 \\ 4 & -4 \end{pmatrix}, \quad N = \begin{pmatrix} -2 & -8 \\ 0 & 4 \end{pmatrix}$$

### (2) 오차 벡터

$$\mathbf{e}^{(k+1)} = \begin{pmatrix} -3 \\ -4 \end{pmatrix}$$

**참고**: 오차가 커지고 있으므로 이 설정($\omega=2$)에서 SOR은 발산합니다.

---

**작성일**: 2025-12-03
**과목**: 금융수치해석기법 기말고사 2024
