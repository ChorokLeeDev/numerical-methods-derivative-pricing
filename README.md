# Quant - 금융공학 연구 & 수치해석 라이브러리

금융공학 연구 논문과 파생상품 가격결정 수치해석 Python 구현체입니다.

## 📄 Active Research

| Paper | Venue | Deadline | Status |
|-------|-------|----------|--------|
| **CW-ACI for Factor Return Uncertainty** | JoFE Special Issue | Mar 1, 2026 | 📄 제출 준비 |

### CW-ACI (Crowding-Weighted Adaptive Conformal Inference)

혼잡도 가중 적응형 Conformal Prediction으로 팩터 수익률 불확실성 정량화.

**핵심 결과:**
- High-crowding 커버리지: 75% → 95% (+19pp)
- VaR breach 83% 감소
- FF 팩터, 암호화폐, 섹터 ETF 등 다양한 자산군에서 검증

📁 [research/jofe_cwaci/](research/jofe_cwaci/) | 📄 [Paper PDF](research/jofe_cwaci/paper/main.pdf)

---

## 📂 저장소 구조

```
quant/
├── README.md
├── research/
│   ├── jofe_cwaci/           # ⭐ 활성 논문
│   └── ARCHIVED_PAPERS.md    # 폐기 기록
├── archive/                   # 폐기된 프로젝트들
├── quant/                     # 코어 라이브러리
└── tests/
```

## 주요 기능

### 옵션 가격결정
- Black-Scholes 해석해
- QuantLib 연동 (바닐라, 배리어, 바스켓)
- FDM (Explicit/Implicit/Crank-Nicolson)
- American Option PSOR

### 수치해석
- 선형방정식 (LU, Cholesky, QR, SVD)
- 반복해법 (Gauss-Seidel, SOR)

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 라이선스

Educational Purpose
