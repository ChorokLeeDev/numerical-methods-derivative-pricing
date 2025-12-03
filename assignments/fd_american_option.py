#%%
import numpy as np 
import pandas as pd
from scipy.linalg import solve_banded
from scipy.interpolate import interp1d

def fd_american_option(s, k, r, q, t, sigma, option_type, n, m):
    """
    PSOR (Projected Successive Over-Relaxation) 방법을 사용한 아메리칸 플레인-바닐라 옵션 가격 계산

    교육용 배경 지식:
    ================

    1. 아메리칸 옵션:
       - 만기 이전 언제든지 행사 가능 (유럽형 옵션과 다름)
       - 조기 행사 기능으로 인해 유럽형 옵션보다 가치가 높음
       - 각 시점마다 조기 행사가 최적인지 확인 필요

    2. 유한차분법 (FDM):
       - Black-Scholes PDE를 그리드로 이산화
       - 그리드 점: (S_i, t_j) - S_i는 자산가격, t_j는 시간
       - 만기에서 현재로 역방향 시간 진행

    3. PSOR (투영 SOR):
       - SOR = Successive Over-Relaxation (선형방정식을 푸는 반복법)
       - "Projected"는 해를 제약조건 집합으로 투영한다는 의미
       - 제약조건: V(S,t) >= Payoff(S) (옵션가치 >= 내재가치)
       - 아메리칸 옵션의 자유경계 문제를 처리
       - Omega (ω)는 완화 파라미터 (일반적으로 1.0~1.2)

    매개변수:
    ---------
    s : float - 현재 기초자산 가격
    k : float - 행사가격
    r : float - 무위험 이자율 (연율)
    q : float - 연속 배당률 (연율)
    t : float - 잔존만기 (년)
    sigma : float - 변동성 (연율)
    option_type : str - "call" 또는 "put"
    n : int - 자산가격 스텝 수
    m : int - 시간 스텝 수

    반환값:
    -------
    tuple: (price, delta, gamma, theta)
        - price: 현재 자산가격에서의 옵션 가치
        - delta: ∂V/∂S (자산가격 변화에 대한 민감도)
        - gamma: ∂²V/∂S² (볼록성, 델타 변화율)
        - theta: ∂V/∂t (시간 감소, 일일 기준)
    """

    # ===========================================
    # 단계 1: 그리드 설정
    # ===========================================

    # 최대 자산가격 설정 (현재가격의 4배면 충분)
    s_max = s * 4

    # 그리드 간격 계산
    ds = s_max / n  # 공간 스텝 (자산가격 증가분)
    dt = t / m      # 시간 스텝 (시간 증가분)

    # 콜/풋 옵션 구분 (콜=1, 풋=-1)
    call_or_put = 1 if option_type.lower() == 'call' else -1

    # 자산가격 그리드 생성: [0, ds, 2*ds, ..., n*ds = s_max]
    i = np.arange(n + 1)
    S = i * ds

    # ===========================================
    # 단계 2: 유한차분 계수 계산
    # ===========================================
    # ** Black-Scholes PDE란? **
    # - PDE = Partial Differential Equation (편미분방정식)
    # - Black-Scholes 방정식: 옵션 가격을 지배하는 편미분방정식
    # - 주식 가격이 기하 브라운 운동을 따른다고 가정
    # - 무차익 조건(no-arbitrage)으로부터 도출됨
    #
    # Black-Scholes PDE:
    # ∂V/∂t + (r-q)S∂V/∂S + 0.5σ²S²∂²V/∂S² - rV = 0
    #
    # 각 항의 의미:
    # - ∂V/∂t: 시간에 따른 옵션가격 변화
    # - (r-q)S∂V/∂S: 자산 가격의 드리프트(추세) 효과
    # - 0.5σ²S²∂²V/∂S²: 자산 가격의 변동성(확산) 효과
    # - rV: 할인 효과 (무위험 이자율)

    # 내부 격자점 (i = 1, 2, ..., n-1)에 대해:
    a = dt * (sigma * S[1:-1])**2 / (2 * ds**2)  # V[i-1] 계수 (2차 미분항)
    b = dt * (r - q) * S[1:-1] / (2 * ds)         # 드리프트항 계수

    # Crank-Nicolson 계수 (theta = 0.5)
    # d: 하부대각선, mid: 주대각선, u: 상부대각선
    d = a - b      # V[i-1] 계수
    mid = -2 * a - dt * r  # V[i] 계수
    u = a + b      # V[i+1] 계수

    # ===========================================
    # 단계 3: Crank-Nicolson 행렬 구성
    # ===========================================
    # ** Crank-Nicolson 방법이란? **
    # - 시간 방향으로 PDE를 이산화하는 방법 중 하나
    # - 암시적(implicit) 방법과 명시적(explicit) 방법의 평균 (theta = 0.5)
    #
    # 장점:
    # - 무조건 안정적(unconditionally stable) - 큰 시간 스텝 사용 가능
    # - 2차 정확도(second-order accurate) - 오차가 작음
    #
    # 형태: (I - 0.5*A)V^{j} = (I + 0.5*A)V^{j+1} + 경계항
    # - 좌변: 현재 시점 j (암시적 - 행렬 방정식을 풀어야 함)
    # - 우변: 다음 시점 j+1 (명시적 - 이미 알고 있는 값)

    # 삼중대각 행렬 A 구성
    A = np.diag(d[1:], -1) + np.diag(mid) + np.diag(u[:-1], 1)

    # 경계 기여 행렬
    B = np.zeros((n - 1, 2))
    B[0, 0], B[-1, 1] = d[0], u[-1]

    # 좌변 행렬: I - 0.5*A
    theta_param = 0.5  # Crank-Nicolson은 theta = 0.5 사용
    Am = np.identity(n - 1) - theta_param * A

    # ** 삼중대각 행렬(Tridiagonal Matrix)이란? **
    # - 주대각선과 그 위아래 대각선에만 0이 아닌 값이 있는 행렬
    # - 유한차분법으로 1차원 PDE를 이산화하면 자연스럽게 나타남
    # - 일반 행렬보다 훨씬 효율적으로 풀 수 있음 (O(n) vs O(n³))
    #
    # 예시 (5x5):
    # [b₁ c₁  0  0  0]
    # [a₁ b₂ c₂  0  0]
    # [ 0 a₂ b₃ c₃  0]
    # [ 0  0 a₃ b₄ c₄]
    # [ 0  0  0 a₄ b₅]
    #
    # 효율적 계산을 위해 대역 형식(banded format)으로 변환
    # ab[0,:] = 상부대각선, ab[1,:] = 주대각선, ab[2,:] = 하부대각선
    ab = np.zeros((3, n - 1))
    ab[0, 1:] = np.diag(Am, 1)
    ab[1] = np.diag(Am)
    ab[2, :-1] = np.diag(Am, -1)

    # ===========================================
    # 단계 4: 만기시점 페이오프 초기화
    # ===========================================
    # 만기 (t = T)에서 옵션가치 = 내재가치 (intrinsic value)
    #
    # ** 옵션 페이오프 설명 **
    # S = 만기시점의 기초자산 가격 (Stock price at maturity)
    # K = 행사가격 (Strike price) = k 변수
    #
    # 콜옵션 (Call): 자산을 K원에 살 권리
    #   - S > K일 때: K원에 사서 S원에 팔면 이익 = S - K
    #   - S ≤ K일 때: 행사 안 함 (손해), 가치 = 0
    #   - 페이오프 = max(S - K, 0)
    #
    # 풋옵션 (Put): 자산을 K원에 팔 권리
    #   - S < K일 때: S원에 사서 K원에 팔면 이익 = K - S
    #   - S ≥ K일 때: 행사 안 함 (손해), 가치 = 0
    #   - 페이오프 = max(K - S, 0)
    #
    # ** 코드 구현 설명 **
    # call_or_put = 1 (콜) 또는 -1 (풋)
    #
    # 콜일 때 (call_or_put = 1):
    #   v = max(1 * (S - k), 0) = max(S - K, 0) ✓
    #
    # 풋일 때 (call_or_put = -1):
    #   v = max(-1 * (S - k), 0) = max(-(S - K), 0) = max(K - S, 0) ✓
    #
    # 이 방법으로 한 줄의 코드로 콜/풋 둘 다 처리 가능!

    v = np.maximum(call_or_put * (S - k), 0)
    payoff = v.copy()  # 조기행사 비교를 위해 페이오프 저장

    # ===========================================
    # 단계 5: PSOR을 사용한 역방향 귀납법
    # ===========================================
    # ** 역방향 귀납법(Backward Induction)이란? **
    # - 시간을 거슬러 올라가며 계산하는 방법
    # - 옵션 가격 계산 순서:
    #   1. 만기(t=T): 페이오프로 초기화 (확실히 알고 있음)
    #   2. t=T-dt: 만기 값을 이용해 역산
    #   3. t=T-2dt: 앞 단계 값을 이용해 역산
    #   ...
    #   N. t=0 (현재): 최종 옵션 가격
    #
    # 만기에서 현재까지 역방향으로 계산

    omega = 1.2  # 과완화 파라미터 (수렴 속도 향상)
    tol = 1e-6   # 수렴 허용오차
    max_iter = 1000  # 시간 스텝당 최대 PSOR 반복 횟수

    for j in range(m - 1, -1, -1):
        # ========================================
        # 단계 5.1: 경계 조건 설정
        # ========================================
        # S=0일 때: 콜은 0, 풋은 K*exp(-r*τ)
        # S=S_max일 때: 콜은 S_max - K*exp(-r*τ), 풋은 0

        tau = (m - j) * dt  # 현재 스텝에서 만기까지 시간

        if option_type.lower() == 'call':
            v[0] = 0
            v[n] = s_max - k * np.exp(-r * tau)
        else:  # put
            v[0] = k * np.exp(-r * tau)
            v[n] = 0

        # ========================================
        # 단계 5.2: 우변 계산
        # ========================================
        # Crank-Nicolson 기법의 우변
        temp = (1 - theta_param) * d * v[:-2] + \
               (1 + (1 - theta_param) * mid) * v[1:-1] + \
               (1 - theta_param) * u * v[2:]
        temp[0] += theta_param * d[0] * v[0]
        temp[-1] += theta_param * u[-1] * v[n]
        temp += (1 - theta_param) * B @ v[[0, -1]]

        # ========================================
        # 단계 5.3: PSOR 반복
        # ========================================
        # 조기행사 제약조건과 함께 반복적으로 해 계산
        # Gauss-Seidel 반복법과 투영 사용

        v_new = v[1:-1].copy()

        for iteration in range(max_iter):
            v_old_iter = v_new.copy()

            # Gauss-Seidel SOR 스윕
            for ii in range(n - 1):
                # 제약조건 없이 값 계산
                if ii == 0:
                    v_temp = (temp[ii] - ab[0, 1] * v_new[ii + 1]) / ab[1, ii]
                elif ii == n - 2:
                    v_temp = (temp[ii] - ab[2, ii - 1] * v_new[ii - 1]) / ab[1, ii]
                else:
                    v_temp = (temp[ii] - ab[2, ii - 1] * v_new[ii - 1] - ab[0, ii + 1] * v_new[ii + 1]) / ab[1, ii]

                # SOR 업데이트
                v_temp = v_new[ii] + omega * (v_temp - v_new[ii])

                # 투영: 조기행사 제약조건 적용
                # 옵션 가치는 내재가치 이상이어야 함
                v_new[ii] = max(v_temp, payoff[ii + 1])

            # 수렴 확인
            if np.max(np.abs(v_new - v_old_iter)) < tol:
                break

        v[1:-1] = v_new

        # 세타 계산을 위해 한 시간 스텝 전 값 저장
        if j == 1:
            f = interp1d(S, v, kind='cubic')
            p1 = f(s)

    # ===========================================
    # 단계 6: 그릭스 계산
    # ===========================================
    # ** 그릭스(Greeks)란? **
    # 옵션 가격이 여러 변수에 얼마나 민감한지 측정하는 지표들
    # 그리스 문자로 표기되어 "Greeks"라고 부름
    #
    # 주요 그릭스:
    # - Delta (Δ): 자산가격이 1원 변할 때 옵션가격 변화량
    #   예: Delta = 0.5 → 주가 1원 ↑ 시 옵션가 0.5원 ↑
    #
    # - Gamma (Γ): 자산가격이 1원 변할 때 Delta의 변화량 (델타의 변화율)
    #   예: Gamma = 0.02 → 주가 1원 ↑ 시 Delta가 0.02 증가
    #   볼록성(convexity) 측정 - 위험관리에 중요
    #
    # - Theta (Θ): 시간이 하루 지날 때 옵션가격 변화량 (시간 감소)
    #   예: Theta = -0.05 → 하루 지나면 옵션가 0.05원 감소
    #   "시간 가치 붕괴(time decay)" 측정
    #
    # - Vega (ν): 변동성이 1% 변할 때 옵션가격 변화량 (코드에는 없음)
    # - Rho (ρ): 이자율이 1% 변할 때 옵션가격 변화량 (코드에는 없음)
    #
    # ** 보간법(Interpolation)이란? **
    # - 그리드 점들 사이의 값을 추정하는 방법
    # - 우리는 이산화된 그리드 점에서만 옵션 가격을 계산
    # - 하지만 실제 자산가격 s는 그리드 점이 아닐 수 있음
    # - 보간법으로 정확한 s에서의 옵션 가격을 추정
    #
    # Cubic Spline 보간:
    # - 3차 다항식을 이용한 부드러운 곡선 피팅
    # - 1차 및 2차 도함수도 연속적 (Greeks 계산에 유리)
    #
    # 보간법과 유한차분 사용하여 계산

    f = interp1d(S, v, kind='cubic')

    # 현재 스팟에서의 가격
    price = f(s)

    # 델타: S에 대한 1차 미분
    # ** 중앙차분(Central Difference)이란? **
    # - 도함수를 근사하는 수치 미분 방법
    # - 전진차분: [f(x+h) - f(x)] / h (1차 정확도)
    # - 후진차분: [f(x) - f(x-h)] / h (1차 정확도)
    # - 중앙차분: [f(x+h) - f(x-h)] / (2h) (2차 정확도, 더 정확!)
    #
    # 중앙차분 사용: δV/δS ≈ [V(S+h) - V(S-h)] / (2h)
    h = 0.01 * s
    delta = (f(s + h) - f(s - h)) / (2 * h)

    # 감마: S에 대한 2차 미분
    # 중앙차분 사용: δ²V/δS² ≈ [V(S+h) - 2V(S) + V(S-h)] / h²
    gamma = (f(s + h) - 2 * price + f(s - h)) / (h**2)

    # 세타: 시간 감소 (일일 기준)
    # 현재 시점과 한 시간 스텝 전 값 비교
    theta = (p1 - price) / dt / 365.0

    return price, delta, gamma, theta

if __name__=="__main__":
    s = 100
    k = 100
    r = 0.03
    q = 0.01
    t = 1
    sigma = 0.3
    optionType = 'put'
    n, m = 500, 500
    print("="*50)
    print("American Option")
    print("="*50)

    price, delta, gamma, theta = fd_american_option(s, k, r, q, t, sigma, optionType, n, m)
    print(f"American(CN-FDM) = {price:0.6f}")




# %%
