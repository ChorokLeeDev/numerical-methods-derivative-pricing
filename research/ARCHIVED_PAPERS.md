# Archived Papers

폐기된 논문들의 기록. 모든 폐기 프로젝트는 `archive/` 폴더에 있습니다.

| Paper | Target | 사유 | 위치 |
|-------|--------|------|------|
| ICML CW-ACI | ICML 2026 | JoFE로 통합 | `archive/icml2026_conformal/` |
| KDD Global Crowding | KDD 2026 | Temporal-MMD 실패 | `archive/kdd2026_global_crowding/` |
| JMLR Unified | JMLR | 게임이론 순환논리 | `archive/jmlr_unified/` |

---

## 1. ICML → JoFE 통합

동일한 CW-ACI 방법론. 중복 제거를 위해 JoFE로 병합.

병합된 내용: Proposition 3개 (Coverage, Uniformity, Regret Bound)

---

## 2. KDD - Temporal-MMD 실패

| Transfer | 결과 |
|----------|------|
| US→Japan | +18.9% ✓ |
| US→Europe | **-21.5%** ✗ |
| 평균 | **-5.2%** ✗ |

근본 원인: "Domain-invariant regimes" 가정 실패.

---

## 3. JMLR - 다중 실패

| Component | 문제 |
|-----------|------|
| Game Theory | 순환 논리 |
| MMD Transfer | -318% |
| **CW-ACI** | **성공 → JoFE로 분리** |

---

*Last updated: December 23, 2025*
