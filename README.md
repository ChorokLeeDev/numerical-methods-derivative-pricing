# Quant - 금융공학 연구 & 수치해석 라이브러리

금융공학 연구 논문과 파생상품 가격결정 수치해석 Python 구현체입니다.

> **⚠️ STATUS**: 3개 진행 중인 논문 + 1개 코어 라이브러리. 최근 디버그 조사(2025-12-16) 완료.
> 자세한 내용은 [research/PROJECT_OVERVIEW.md](research/PROJECT_OVERVIEW.md) 참조.

## Research Papers (3개 진행 중)

### 1. JMLR 통합 프레임워크 (주요 논문)
**"Not All Factors Crowd Equally: Unified Framework"** 🟡 **재구성 필요**
- 성분 1: 게임이론 alpha 감소 모델 (Theorem 5)
- 성분 2: 레짐-조건부 도메인 적응 (Temporal-MMD) **문제 발견**
- 성분 3: Conformal prediction 위험 관리
- **상태**: 코드 정확, 경험적 주장 문제 (재구성 권장)
- [상세 정보](research/PROJECT_DETAILS.md)

### 2. KDD 2026 (Jeju, Korea) - 마감 2월 8일
**"Mining Factor Crowding at Global Scale"** 🔴 **디버그 완료**
- 6개 지역 × 10+ 팩터 = 60개 팩터-지역 쌍
- ML 탐지 (LSTM/XGBoost) vs 모델 기반 비교
- **문제**: Temporal-MMD이 Europe에서 -21.5% 성능 저하 (Japan은 +18.9%)
- **원인**: 레짐 정의는 시장별 특이적, 보편적이지 않음
- [디버그 보고서](research/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md) ⭐
- [상세 분석](research/kdd2026_global_crowding/experiments/DIAGNOSTIC_REPORT.md)

### 3. ICML 2026 (Seoul, Korea) - 마감 1월 28일
**"Conformal Prediction for Factor Crowding"** 🟢 **진행 중**
- Distribution-free 불확실성 정량화
- 커버리지 보장 있는 예측 집합
- Bayesian/Bootstrap 방법 비교
- [상세 정보](research/icml2026_conformal/README.md)

## 📂 저장소 구조

```
quant/
├── README.md ← 현재 파일
├── research/ ← 3개 활성 논문 프로젝트
│   ├── PROJECT_OVERVIEW.md ⭐ (읽어야 할 파일)
│   ├── PROJECT_DETAILS.md  ⭐ (상세 정보)
│   ├── jmlr_unified/ (JMLR 논문 - 재구성 필요)
│   ├── kdd2026_global_crowding/ (KDD 논문 - 디버그 완료)
│   │   └── experiments/
│   │       ├── FINAL_SUMMARY.md ⭐ (꼭 읽기)
│   │       ├── DIAGNOSTIC_REPORT.md
│   │       └── 09-13_*.py (재현 가능한 진단 스크립트)
│   └── icml2026_conformal/ (ICML 논문 - 진행 중)
├── quant/ ← 코어 라이브러리
│   ├── factors/
│   ├── ml/
│   ├── portfolio/
│   ├── pricing/
│   ├── data/
│   └── numerical/
├── tests/ ← 테스트 (분산됨, 정리 필요)
├── notebooks/ ← Jupyter (정리 필요)
├── data/ ← 데이터 및 캐시
└── docs/ ← 문서
```

## 🎯 지금 해야 할 것 (긴급도순)

### 📖 먼저 읽어야 할 것 (이번 주)
1. **research/PROJECT_OVERVIEW.md** - 프로젝트 상황 파악 (5분)
2. **research/PROJECT_DETAILS.md** - 각 프로젝트 상세 (10분)
3. **research/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md** - 디버그 결과 (10분)

### 🔧 해야 할 결정 (이번 주)
- [ ] JMLR 논문: jmlr_unified와 factor-crowding-unified 통합?
- [ ] KDD 논문: Temporal-MMD 유지 (조건부) vs 제거 vs 대체?
- [ ] 저장소 구조: 풀 재구성 vs 최소 정리?

### ⏱️ 일정
- ICML 2026: 1월 28일 (≈6주)
- KDD 2026: 2월 8일 (≈7주)
- JMLR: 언제든지

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

### 금융 인수분해 (Factor Modeling)
- Fama-French 8개 팩터 (1963-2025)
- 글로벌 6개 지역 (US, UK, Japan, Europe, AsiaPac)
- 팩터 혼잡도 탐지 (Crowding Detection)

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

## 📚 추가 문서

**CRITICAL (꼭 읽기)**:
- [research/PROJECT_OVERVIEW.md](research/PROJECT_OVERVIEW.md) - 전체 프로젝트 현황
- [research/PROJECT_DETAILS.md](research/PROJECT_DETAILS.md) - 각 프로젝트 상세
- [research/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md](research/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md) - 디버그 조사 결과

**프로젝트별**:
- [research/kdd2026_global_crowding/experiments/README_DIAGNOSTIC_SESSION.md](research/kdd2026_global_crowding/experiments/README_DIAGNOSTIC_SESSION.md) - 진단 세션 가이드
- [research/kdd2026_global_crowding/experiments/DEBUG_SESSION_CLEANUP.md](research/kdd2026_global_crowding/experiments/DEBUG_SESSION_CLEANUP.md) - 정리 로그

**개발**:
- [INDEX.md](INDEX.md) - 모듈 상세 설명 (있는 경우)

## 참고 문헌

- Stulz (1982) - Options on the Minimum or Maximum of Two Risky Assets
- Hull - Options, Futures, and Other Derivatives
- Wilmott - Paul Wilmott on Quantitative Finance

## 최근 업데이트

- **2025-12-16**: Option D 디버그 조사 완료 - Temporal-MMD 문제 근본원인 규명
  - Regime 정의는 시장별 특이적 (domain-specific)
  - 도메인 불변(domain-invariant) 가정 위반
  - Europe에서 -21.5% 성능 저하, Japan에서 +18.9% 성공
  - 상세: [research/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md](research/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md)

## 라이선스

Educational Purpose

---

**🔴 NEXT STEPS**: 상단의 "지금 해야 할 것" 섹션을 읽고 진행하세요.
