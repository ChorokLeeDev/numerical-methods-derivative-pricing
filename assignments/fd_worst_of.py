
#%%
import QuantLib as ql
import numpy as np
from src.pricing.ql_worst_of import ql_worst_of

def osm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt):
    """
    OSM (Operator Splitting Method)을 사용한 오버헤지 파라미터가 있는 최악자산 이진옵션 가격 계산

    교육용 배경 지식:
    ================

    1. OPERATOR SPLITTING METHOD (OSM):
       - "부분 스텝 방법"으로도 알려짐
       - 2차원 PDE를 더 간단한 1차원 문제로 분할
       - ADI와 유사하지만 다른 분할 전략 사용
       - 강한 교차미분이 있는 문제에서 더 정확

    2. ADI와의 주요 차이점:
       - ADI: S1과 S2 방향을 반스텝씩 번갈아가며 진행
       - OSM: 각 방향을 풀스텝으로 순차적으로 해결
       - OSM은 교차미분을 더 명시적으로 처리

    3. 분할 방식:
       - 단계 1: 순수 S1 방향 해결 (S1 미분만 포함)
       - 단계 2: 순수 S2 방향 해결 (S2 미분만 포함)
       - 단계 3: 교차미분 보정 적용
       - 각 연산자를 별도로 해결한 후 결합

    매개변수:
    ---------
    s1, s2 : float - 자산 1과 2의 현재 가격
    k : float - 행사가격
    r : float - 무위험 이자율
    q1, q2 : float - 자산 1과 2의 배당 수익률
    t : float - 잔존만기
    sigma1, sigma2 : float - 변동성
    corr : float - 상관계수
    oh : float - 오버헤지 파라미터
    nx, ny : int - S1과 S2의 스텝 수
    nt : int - 시간 스텝 수

    반환값:
    -------
    tuple: (price, delta1, delta2, gamma1, gamma2, cross_gamma, theta)
    """

    # ===========================================
    # 단계 1: 그리드 설정
    # ===========================================

    s1_max = s1 * 2
    s2_max = s2 * 2

    dx = s1_max / nx
    dy = s2_max / ny
    dt = t / nt

    # ** 2D 그리드(Grid)란? **
    # - 2차원 공간을 격자점으로 나누는 것
    # - 1차원: 선을 점들로 나눔 [0, dx, 2dx, ..., S_max]
    # - 2차원: 평면을 격자점들로 나눔 (S1, S2) 좌표
    #
    # meshgrid: 1차원 배열 2개를 2차원 그리드로 확장
    # - x = [0, 50, 100], y = [0, 50, 100]
    # - X, Y = meshgrid(x, y)로 모든 (x, y) 조합 생성
    # - indexing='ij': 행렬 인덱스 방식 (i=S1, j=S2)
    #
    # 2D 그리드 생성
    x = np.linspace(0, s1_max, nx + 1)
    y = np.linspace(0, s2_max, ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # ===========================================
    # 단계 2: 페이오프 초기화
    # ===========================================
    # ** 최악자산 이진옵션 페이오프 **
    # - 이진(Binary/Digital) 옵션: 조건 만족 시 고정금액, 아니면 0
    # - 최악자산(Worst-of): 두 자산 중 낮은 가격 기준
    #
    # 페이오프 = 10,000원 (min(S1, S2) > K일 때)
    #         = 0원 (그 외)
    #
    # 예시: S1=110, S2=90, K=95
    # → min(110, 90) = 90 < 95 → 페이오프 = 0
    #
    # 예시: S1=110, S2=100, K=95
    # → min(110, 100) = 100 > 95 → 페이오프 = 10,000

    k_modified = k * (1 - oh / 10000)
    payoff_value = 10000.0

    V = np.where(np.minimum(X, Y) > k_modified, payoff_value, 0.0)

    # ===========================================
    # 단계 3: 연산자 준비
    # ===========================================
    # OSM은 세 개의 연산자로 분할:
    # L1: S1 방향, L2: S2 방향, L12: 교차미분

    # S1 연산자 계수
    def build_L1_coefficients(dt_step):
        a1 = np.zeros(nx + 1)
        b1 = np.zeros(nx + 1)
        c1 = np.zeros(nx + 1)

        for i in range(1, nx):
            s1_i = i * dx
            a1[i] = 0.5 * dt_step * ((sigma1 * s1_i / dx)**2 - (r - q1) * s1_i / dx)
            c1[i] = 0.5 * dt_step * ((sigma1 * s1_i / dx)**2 + (r - q1) * s1_i / dx)
            b1[i] = -dt_step * ((sigma1 * s1_i / dx)**2 + r)

        return a1, b1, c1

    # S2 연산자 계수
    def build_L2_coefficients(dt_step):
        a2 = np.zeros(ny + 1)
        b2 = np.zeros(ny + 1)
        c2 = np.zeros(ny + 1)

        for j in range(1, ny):
            s2_j = j * dy
            a2[j] = 0.5 * dt_step * ((sigma2 * s2_j / dy)**2 - (r - q2) * s2_j / dy)
            c2[j] = 0.5 * dt_step * ((sigma2 * s2_j / dy)**2 + (r - q2) * s2_j / dy)
            b2[j] = -dt_step * ((sigma2 * s2_j / dy)**2 + r)

        return a2, b2, c2

    a1, b1, c1 = build_L1_coefficients(dt)
    a2, b2, c2 = build_L2_coefficients(dt)

    # 교차미분 계수
    rho = corr

    # ===========================================
    # 단계 4: OSM을 사용한 역방향 시간 진행
    # ===========================================

    V_star = np.zeros_like(V)   # S1 단계 후
    V_star2 = np.zeros_like(V)  # S2 단계 후

    for n in range(nt - 1, -1, -1):
        # ========================================
        # 단계 4.1: L1 연산자 해결 (S1 방향)
        # ========================================
        # ∂V/∂t + L1(V) = 0 해결, 여기서 L1은 S1 미분 포함

        for j in range(ny + 1):
            # ** OSM의 차원 분리(Dimension Splitting) 개념 **
            # - 2D 문제를 1D 문제들로 나눔
            # - 여기서는 S2를 고정하고 S1 방향으로만 PDE 풀기
            # - 각 S2=j 값마다 독립적인 1D 문제
            # - V[:, j]는 S2=j일 때 모든 S1 값에서의 옵션 가격
            #
            # S2=j로 고정한 S1에 대한 삼중대각 시스템 구축
            rhs = V[:, j].copy()

            # 시스템 행렬 구축
            lower = -a1[1:nx]
            main = 1 - b1[1:nx]
            upper = -c1[1:nx]

            # 경계 조건 (경계에서 옵션 가치 없음)
            rhs[1] -= a1[1] * 0
            rhs[nx - 1] -= c1[nx - 1] * 0

            # 삼중대각 시스템 해결
            from scipy.linalg import solve_banded
            ab = np.zeros((3, nx - 1))
            ab[0, 1:] = upper[1:]
            ab[1, :] = main
            ab[2, :-1] = lower[:-1]

            V_star[1:nx, j] = solve_banded((1, 1), ab, rhs[1:nx])
            V_star[0, j] = 0
            V_star[nx, j] = 0

        # ========================================
        # 단계 4.2: L2 연산자 해결 (S2 방향)
        # ========================================
        # ∂V/∂t + L2(V) = 0 해결, 여기서 L2는 S2 미분 포함

        for i in range(nx + 1):
            # S1=i로 고정한 S2에 대한 삼중대각 시스템 구축
            rhs = V_star[i, :].copy()

            # 시스템 행렬 구축
            lower = -a2[1:ny]
            main = 1 - b2[1:ny]
            upper = -c2[1:ny]

            # 경계 조건
            rhs[1] -= a2[1] * 0
            rhs[ny - 1] -= c2[ny - 1] * 0

            # 삼중대각 시스템 해결
            ab = np.zeros((3, ny - 1))
            ab[0, 1:] = upper[1:]
            ab[1, :] = main
            ab[2, :-1] = lower[:-1]

            V_star2[i, 1:ny] = solve_banded((1, 1), ab, rhs[1:ny])
            V_star2[i, 0] = 0
            V_star2[i, ny] = 0

        # ========================================
        # 단계 4.3: 교차미분 보정 적용
        # ========================================
        # 혼합 미분항 추가: ρ σ1 σ2 S1 S2 ∂²V/∂S1∂S2

        V_temp = V_star2.copy()

        for i in range(1, nx):
            for j in range(1, ny):
                s1_i = i * dx
                s2_j = j * dy

                # ** 교차미분(Cross Derivative)이란? **
                # - 두 변수에 대해 동시에 미분: ∂²V/∂S1∂S2
                # - 두 자산 가격의 상호작용 효과
                # - 상관계수(correlation)와 관련
                #
                # 왜 필요한가?
                # - 2자산 Black-Scholes PDE의 항: ρσ1σ2 S1 S2 ∂²V/∂S1∂S2
                # - ρ(상관계수)가 0이 아니면 두 자산이 함께 움직임
                #
                # 중앙차분을 사용한 교차미분 근사
                # ∂²V/∂S1∂S2 ≈ [V(i+1,j+1) - V(i+1,j-1) - V(i-1,j+1) + V(i-1,j-1)] / (4*dx*dy)
                cross_deriv = (V[i + 1, j + 1] - V[i + 1, j - 1] -
                              V[i - 1, j + 1] + V[i - 1, j - 1]) / (4 * dx * dy)

                # 보정 적용
                correction = dt * rho * sigma1 * sigma2 * s1_i * s2_j * cross_deriv
                V_temp[i, j] += correction

        V = V_temp.copy()

        # 세타 계산을 위해 저장
        if n == 1:
            V_prev = V.copy()

    # ===========================================
    # 단계 5: 그릭스 계산
    # ===========================================
    # ** 그릭스(Greeks) **
    # - 옵션 가격의 민감도 지표들
    # - 위험 관리 및 헤징에 필수
    #
    # 2자산 옵션의 그릭스:
    # - Delta1, Delta2: 각 자산 가격에 대한 민감도
    # - Gamma1, Gamma2: 각 자산의 델타 변화율 (볼록성)
    # - Cross-Gamma: ∂²V/∂S1∂S2 (두 자산 간 상호작용)
    # - Theta: 시간 감소
    #
    # ** RectBivariateSpline이란? **
    # - 2차원 보간법 (Rectangular grid에서 Bivariate Spline)
    # - 그리드 점 사이의 값을 부드럽게 추정
    # - 1차 및 2차 도함수도 계산 가능

    from scipy.interpolate import RectBivariateSpline
    spline = RectBivariateSpline(x, y, V)

    price = float(spline(s1, s2)[0, 0])

    # Delta1: ∂V/∂S1
    h1 = 0.01 * s1
    delta1 = (spline(s1 + h1, s2)[0, 0] - spline(s1 - h1, s2)[0, 0]) / (2 * h1)

    # Delta2: ∂V/∂S2
    h2 = 0.01 * s2
    delta2 = (spline(s1, s2 + h2)[0, 0] - spline(s1, s2 - h2)[0, 0]) / (2 * h2)

    # Gamma1: ∂²V/∂S1²
    gamma1 = (spline(s1 + h1, s2)[0, 0] - 2 * price + spline(s1 - h1, s2)[0, 0]) / (h1**2)

    # Gamma2: ∂²V/∂S2²
    gamma2 = (spline(s1, s2 + h2)[0, 0] - 2 * price + spline(s1, s2 - h2)[0, 0]) / (h2**2)

    # Cross-gamma: ∂²V/∂S1∂S2
    # ** Cross-Gamma의 의미 **
    # - 한 자산의 델타가 다른 자산 가격 변화에 얼마나 민감한지
    # - Cross-Gamma > 0: 두 자산이 같은 방향으로 움직일 때 유리
    # - Cross-Gamma < 0: 두 자산이 반대 방향으로 움직일 때 유리
    # - 상관관계가 있는 다자산 포트폴리오 헤징에 중요
    crossgamma = (spline(s1 + h1, s2 + h2)[0, 0] - spline(s1 + h1, s2 - h2)[0, 0] -
                  spline(s1 - h1, s2 + h2)[0, 0] + spline(s1 - h1, s2 - h2)[0, 0]) / (4 * h1 * h2)

    # Theta: ∂V/∂t (일일 기준)
    spline_prev = RectBivariateSpline(x, y, V_prev)
    price_prev = float(spline_prev(s1, s2)[0, 0])
    theta = (price_prev - price) / dt / 365.0

    return price, delta1, delta2, gamma1, gamma2, crossgamma, theta

def adi_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt):
    """
    ADI (Alternating Direction Implicit) 방법을 사용한 오버헤지 파라미터가 있는 최악자산 이진옵션 가격 계산

    교육용 배경 지식:
    ================

    1. 최악자산 이진옵션 (WORST-OF BINARY OPTION):
       - 페이오프가 두 자산 중 최소값에 의존
       - 이진(디지털) 페이오프: min(S1, S2) > K이면 10,000, 아니면 0
       - 2차원 문제: 두 자산 가격을 모두 추적 필요

    2. 오버헤지 파라미터 (OH):
       - 행사가 주변에 완충 구역 추가
       - 수정된 행사가: K_modified = K * (1 - oh/10000)
       - 불연속성 근처의 수치 불안정성 방지
       - 페이오프 함수를 평활화

    3. ADI 방법:
       - Alternating Direction Implicit - 2D 문제를 1D 문제로 분할
       - 단계 1: S1 방향으로 암시적 해결 (S2 고정)
       - 단계 2: S2 방향으로 암시적 해결 (S1 고정)
       - 각 단계가 무조건 안정적
       - 전체 2D 시스템을 푸는 것보다 훨씬 효율적

    4. 2자산 Black-Scholes PDE:
       ∂V/∂t + (r-q1)S1∂V/∂S1 + (r-q2)S2∂V/∂S2
       + 0.5σ1²S1²∂²V/∂S1² + 0.5σ2²S2²∂²V/∂S2²
       + ρσ1σ2 S1 S2 ∂²V/∂S1∂S2 - rV = 0

    매개변수:
    ---------
    s1, s2 : float - 자산 1과 2의 현재 가격
    k : float - 행사가격
    r : float - 무위험 이자율
    q1, q2 : float - 자산 1과 2의 배당 수익률
    t : float - 잔존만기
    sigma1, sigma2 : float - 변동성
    corr : float - 상관계수
    oh : float - 오버헤지 파라미터
    nx, ny : int - S1과 S2의 스텝 수
    nt : int - 시간 스텝 수

    반환값:
    -------
    tuple: (price, delta1, delta2, gamma1, gamma2, cross_gamma, theta)
    """

    # ===========================================
    # 단계 1: 그리드 설정
    # ===========================================

    s1_max = s1 * 2
    s2_max = s2 * 2

    dx = s1_max / nx
    dy = s2_max / ny
    dt = t / nt

    # 2D 그리드 생성
    x = np.linspace(0, s1_max, nx + 1)  # S1 값
    y = np.linspace(0, s2_max, ny + 1)  # S2 값
    X, Y = np.meshgrid(x, y, indexing='ij')

    # ===========================================
    # 단계 2: 페이오프 초기화
    # ===========================================
    # 이진옵션: min(S1, S2) > K_modified이면 10,000, 아니면 0

    k_modified = k * (1 - oh / 10000)  # 오버헤지로 행사가 조정
    payoff_value = 10000.0

    V = np.where(np.minimum(X, Y) > k_modified, payoff_value, 0.0)

    # ===========================================
    # 단계 3: 계수 행렬 준비
    # ===========================================

    # S1 방향 (x-스윕)
    # S1에 대한 삼중대각 시스템의 계수
    a1 = np.zeros(nx + 1)
    b1 = np.zeros(nx + 1)
    c1 = np.zeros(nx + 1)

    for i in range(1, nx):
        s1_i = i * dx
        a1[i] = 0.25 * dt * ((sigma1 * s1_i / dx)**2 - (r - q1) * s1_i / dx)
        c1[i] = 0.25 * dt * ((sigma1 * s1_i / dx)**2 + (r - q1) * s1_i / dx)
        b1[i] = -0.5 * dt * ((sigma1 * s1_i / dx)**2 + r)

    # S2 방향 (y-스윕)
    a2 = np.zeros(ny + 1)
    b2 = np.zeros(ny + 1)
    c2 = np.zeros(ny + 1)

    for j in range(1, ny):
        s2_j = j * dy
        a2[j] = 0.25 * dt * ((sigma2 * s2_j / dy)**2 - (r - q2) * s2_j / dy)
        c2[j] = 0.25 * dt * ((sigma2 * s2_j / dy)**2 + (r - q2) * s2_j / dy)
        b2[j] = -0.5 * dt * ((sigma2 * s2_j / dy)**2 + r)

    # 교차미분 계수
    rho = corr
    gamma_coef = 0.25 * dt * rho * sigma1 * sigma2

    # ===========================================
    # 단계 4: ADI를 사용한 역방향 시간 진행
    # ===========================================

    V_half = np.zeros_like(V)  # 중간 해

    for n in range(nt - 1, -1, -1):
        # ========================================
        # 단계 4.1: 첫 번째 반스텝 (S1 방향)
        # ========================================
        # ** ADI의 핵심 아이디어: 방향별 분할 **
        # - 2D 문제를 두 개의 1D 문제로 분할
        # - 반스텝 1: S1 방향 암시적, S2 방향 명시적
        # - 반스텝 2: S2 방향 암시적, S1 방향 명시적
        #
        # 암시적(Implicit) vs 명시적(Explicit):
        # - 명시적: 이미 알고 있는 값 사용 (계산 쉬움, 조건부 안정)
        # - 암시적: 방정식 풀어야 함 (계산 어려움, 무조건 안정)
        #
        # S1에서 암시적, S2에서 명시적으로 해결

        for j in range(ny + 1):
            # 명시적 S2 항과 교차미분을 포함한 우변 구축
            rhs = np.zeros(nx + 1)

            for i in range(1, nx):
                s1_i = i * dx
                s2_j = j * dy

                # 명시적 S2 미분항
                term_s2 = 0
                if j > 0 and j < ny:
                    term_s2 = a2[j] * V[i, j - 1] + (1 + b2[j]) * V[i, j] + c2[j] * V[i, j + 1]
                elif j == 0:
                    term_s2 = (1 + b2[j]) * V[i, j] + c2[j] * V[i, j + 1]
                else:  # j == ny
                    term_s2 = a2[j - 1] * V[i, j - 1] + (1 + b2[j - 1]) * V[i, j]

                # 교차미분항 (명시적)
                cross_term = 0
                if i > 0 and i < nx and j > 0 and j < ny:
                    cross_term = gamma_coef * (s1_i * s2_j / (dx * dy)) * \
                                 (V[i + 1, j + 1] - V[i + 1, j - 1] -
                                  V[i - 1, j + 1] + V[i - 1, j - 1])

                rhs[i] = term_s2 + cross_term

            # S1에 대한 삼중대각 시스템 구축
            lower = -a1[1:nx]
            main = 1 - b1[1:nx]
            upper = -c1[1:nx]

            # 경계 조건
            # S1=0과 S1=S1_max에서 옵션은 가치 없음 (min(S1,S2)가 K 초과 불가)
            rhs[1] -= a1[1] * 0
            rhs[nx - 1] -= c1[nx - 1] * 0

            # 삼중대각 시스템 해결
            from scipy.linalg import solve_banded
            ab = np.zeros((3, nx - 1))
            ab[0, 1:] = upper[1:]
            ab[1, :] = main
            ab[2, :-1] = lower[:-1]

            V_half[1:nx, j] = solve_banded((1, 1), ab, rhs[1:nx])
            V_half[0, j] = 0
            V_half[nx, j] = 0

        # ========================================
        # 단계 4.2: 두 번째 반스텝 (S2 방향)
        # ========================================
        # S2에서 암시적으로 해결, S1 스텝에서 얻은 V_half 사용

        for i in range(nx + 1):
            # 명시적 S1 항을 포함한 우변 구축
            rhs = np.zeros(ny + 1)

            for j in range(1, ny):
                s1_i = i * dx
                s2_j = j * dy

                # V_half로부터 명시적 S1 항
                term_s1 = 0
                if i > 0 and i < nx:
                    term_s1 = a1[i] * V_half[i - 1, j] + (1 + b1[i]) * V_half[i, j] + c1[i] * V_half[i + 1, j]
                elif i == 0:
                    term_s1 = (1 + b1[i]) * V_half[i, j] + c1[i] * V_half[i + 1, j]
                else:  # i == nx
                    term_s1 = a1[i - 1] * V_half[i - 1, j] + (1 + b1[i - 1]) * V_half[i, j]

                # 교차미분 (첫 번째 스텝에 이미 포함)
                rhs[j] = term_s1

            # S2에 대한 삼중대각 시스템 구축
            lower = -a2[1:ny]
            main = 1 - b2[1:ny]
            upper = -c2[1:ny]

            # 경계 조건
            rhs[1] -= a2[1] * 0
            rhs[ny - 1] -= c2[ny - 1] * 0

            # 삼중대각 시스템 해결
            ab = np.zeros((3, ny - 1))
            ab[0, 1:] = upper[1:]
            ab[1, :] = main
            ab[2, :-1] = lower[:-1]

            V[i, 1:ny] = solve_banded((1, 1), ab, rhs[1:ny])
            V[i, 0] = 0
            V[i, ny] = 0

        # 세타 계산을 위해 저장
        if n == 1:
            V_prev = V.copy()

    # ===========================================
    # 단계 5: 그릭스 계산
    # ===========================================

    # (s1, s2)에서 가격을 얻기 위해 보간
    from scipy.interpolate import RectBivariateSpline
    spline = RectBivariateSpline(x, y, V)

    price = float(spline(s1, s2)[0, 0])

    # Delta1: ∂V/∂S1
    h1 = 0.01 * s1
    delta1 = (spline(s1 + h1, s2)[0, 0] - spline(s1 - h1, s2)[0, 0]) / (2 * h1)

    # Delta2: ∂V/∂S2
    h2 = 0.01 * s2
    delta2 = (spline(s1, s2 + h2)[0, 0] - spline(s1, s2 - h2)[0, 0]) / (2 * h2)

    # Gamma1: ∂²V/∂S1²
    gamma1 = (spline(s1 + h1, s2)[0, 0] - 2 * price + spline(s1 - h1, s2)[0, 0]) / (h1**2)

    # Gamma2: ∂²V/∂S2²
    gamma2 = (spline(s1, s2 + h2)[0, 0] - 2 * price + spline(s1, s2 - h2)[0, 0]) / (h2**2)

    # Cross-gamma: ∂²V/∂S1∂S2
    cross_gamma = (spline(s1 + h1, s2 + h2)[0, 0] - spline(s1 + h1, s2 - h2)[0, 0] -
                   spline(s1 - h1, s2 + h2)[0, 0] + spline(s1 - h1, s2 - h2)[0, 0]) / (4 * h1 * h2)

    # Theta: ∂V/∂t (일일 기준)
    spline_prev = RectBivariateSpline(x, y, V_prev)
    price_prev = float(spline_prev(s1, s2)[0, 0])
    theta = (price_prev - price) / dt / 365.0

    return price, delta1, delta2, gamma1, gamma2, cross_gamma, theta


if __name__=="__main__":
    s1, s2 = 100,100
    r = 0.02
    q1, q2 = 0.015, 0.01
    k = 95
    t = 1
    sigma1, sigma2 = 0.25, 0.25
    corr = 0.3
    nx, ny, nt = 100, 100, 4000
    oh = 10

    osm_price = osm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt)
    print("OSM 가격 = ", osm_price[0])

    adi_price = adi_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt)
    print("ADI 가격 = ", adi_price[0])

