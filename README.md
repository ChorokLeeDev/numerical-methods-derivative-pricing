# Quant - 금융공학 수치해석 라이브러리

파생상품 가격결정과 금융 수치해석을 위한 Python 구현체입니다.

## 주요 기능

### 옵션 가격결정 (Option Pricing)
- **Black-Scholes** 해석해 (유럽형 옵션)
- **QuantLib** 연동 (바닐라, 배리어, 바스켓 옵션)
- **Finite Difference Method** (Explicit, Implicit, Crank-Nicolson)
- **American Option** PSOR 알고리즘
- **Worst-of 바스켓 옵션** Operator Splitting Method

### 수치해석 (Numerical Methods)
- 선형방정식 직접해법 (LU, Cholesky, QR, SVD)
- 반복해법 (Gauss-Seidel, SOR)
- 수치미분 오차 분석

## 기술 스택

```
Python 3.11+  |  NumPy  |  SciPy  |  Pandas  |  QuantLib  |  Matplotlib
```

## 빠른 시작

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 테스트 실행
pytest tests/ -v
```

## 프로젝트 구조

```
quant/
├── src/
│   ├── pricing/              # 옵션 가격결정
│   │   ├── blackscholes.py       # Black-Scholes 해석해
│   │   ├── ql_vanilla_option.py  # QuantLib 바닐라 옵션
│   │   └── ql_worst_of.py        # Worst-of 바스켓 (Stulz 모델)
│   └── numerical/            # 수치해석 알고리즘
│       ├── fdm.py                # FDM (Explicit/Implicit/CN)
│       ├── linear_system_direct.py    # 직접해법
│       ├── linear_system_iterative.py # 반복해법
│       └── derivatives_error.py       # 수치미분 오차
├── assignments/              # 고급 구현
│   ├── fd_american_option.py     # American 옵션 (PSOR)
│   └── fd_worst_of.py            # 2D FDM (OSM)
├── docs/                     # 문서
│   ├── lectures/                 # 강의자료
│   ├── references/               # 논문 (Stulz 1982 등)
│   └── course/                   # 과제/시험 해설
└── tests/
```

## 사용 예시

### Black-Scholes 옵션 가격

```python
from src.pricing.blackscholes import bsprice

price = bsprice(
    s=100,      # 현재가
    k=100,      # 행사가
    r=0.03,     # 무위험이자율
    q=0.01,     # 배당률
    t=1.0,      # 만기 (년)
    sigma=0.2,  # 변동성
    optionType='call'
)
```

### FDM 옵션 가격 및 Greeks

```python
from src.numerical.fdm import fdm_vanilla_option

result, price, delta, gamma, theta = fdm_vanilla_option(
    s0=100, k=100, r=0.03, q=0.01,
    t=1.0, vol=0.2, optionType='call',
    maxS=400, N=200, M=500, theta=0.5  # Crank-Nicolson
)
```

### American 옵션 (PSOR)

```python
from assignments.fd_american_option import fd_american_option

price, delta, gamma, theta = fd_american_option(
    s=100, k=100, r=0.03, q=0.02,
    t=1.0, sigma=0.25, option_type='put',
    n=200, m=500
)
```

## 핵심 알고리즘

| 알고리즘 | 용도 | 파일 |
|---------|------|------|
| Black-Scholes | 유럽형 옵션 해석해 | `blackscholes.py` |
| Crank-Nicolson FDM | PDE 기반 옵션 가격결정 | `fdm.py` |
| PSOR | American 옵션 조기행사 | `fd_american_option.py` |
| Operator Splitting | 2D PDE (다자산 옵션) | `fd_worst_of.py` |
| Thomas Algorithm | 삼중대각 행렬 O(n) | `fdm.py` |
| LU/Cholesky | 선형방정식 직접해법 | `linear_system_direct.py` |

## 학습 경로

1. **기초**: `derivatives_error.py` → 수치미분 오차 이해
2. **선형대수**: `linear_system_direct.py` → LU, Cholesky 분해
3. **해석해**: `blackscholes.py` → Black-Scholes 공식
4. **FDM**: `fdm.py` → PDE 이산화, 안정성 조건
5. **고급**: `fd_american_option.py` → 자유경계 문제, PSOR

자세한 파일 설명은 [INDEX.md](INDEX.md) 참조

## 참고 문헌

- Stulz (1982) - Options on the Minimum or Maximum of Two Risky Assets
- Hull - Options, Futures, and Other Derivatives
- Wilmott - Paul Wilmott on Quantitative Finance

## 라이선스

Educational Purpose
