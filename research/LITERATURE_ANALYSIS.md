# Literature Analysis & Novelty Assessment
## December 16, 2025

---

## 🎯 목표

1. **실패한 방법 폐기**: Temporal-MMD 제거 결정
2. **문헌 검토**: 기존 연구와 겹치는 부분 파악
3. **차별성 확인**: 각 논문의 novelty 검증

---

## 📊 현재 3개 논문 구조

### Paper 1: JMLR - "Not All Factors Crowd Equally: Unified Framework"
**3개 통합 성분**:

| 성분 | 제목 | 상태 | 문제 |
|------|------|------|------|
| 1 | 게임이론 Alpha Decay Model | ✅ 완료 | 없음 |
| 2 | Temporal-MMD (Domain Adaptation) | ❌ 실패 | Europe -21.5% |
| 3 | Conformal Prediction | ✅ 완료 | 없음 |

### Paper 2: KDD 2026 - "Mining Factor Crowding at Global Scale"
**ML 기반 탐지**:
- 글로벌 6개 지역 × 10+ 팩터
- LSTM/XGBoost vs Model Residuals
- **문제**: Temporal-MMD 결과 혼합 (Japan 좋음, Europe 나쁨)

### Paper 3: ICML 2026 - "Conformal Prediction for Factor Crowding"
**Conformal 접근**:
- Distribution-free uncertainty quantification
- Coverage guarantees
- **상태**: 독립적, 문제 없음

---

## ❌ Temporal-MMD 폐기 결정

### 문제점
```
이론: 레짐-조건부 MMD가 전이 성능 향상
실제:
  - Japan: +18.9% ✓ (작동)
  - Europe: -21.5% ✗ (심각한 실패)
  - Average: -5.2% ✗ (부정적 전이)

원인: 레짐 정의가 시장별 특이적 (domain-specific)
      도메인 불변(domain-invariant) 가정 위반
```

### 폐기 이유
- ✓ Conditional success (조건부 성공)는 논문에 부적절
- ✗ "consistent improvements across markets" 주장 거짓
- ✗ Novel 하지만 실용성 없음
- ✗ 경험적 검증 실패

### 폐기 후 옵션

**Option A: Standard MMD로 대체** ✅ RECOMMENDED
```
장점:
  - 더 간단하고 이해하기 쉬움
  - 더 강건함 (robustness)
  - 여전히 개선 효과 있음 (Europe: 0.608 vs RF: 0.572)

단점:
  - Novelty 감소
  - 이미 알려진 방법

결론: 실용성 > 학술적 novelty
```

**Option B: 도메인 적응 완전 제거**
```
대신:
  - 각 시장별 독립적 모델 훈련
  - 지역별 특성 강조
  - Transfer 주장 하지 않기
```

### 추천 (사용자 선택 필요)
**Option A**: Standard MMD로 대체
- JMLR: Temporal-MMD 제거, Standard MMD만 포함
- KDD: Temporal-MMD 제거, Standard MMD만 포함
- ICML: 변경 없음 (이미 독립적)

---

## 📚 Literature Review & Novelty Check

### Topic 1: Factor Crowding & Alpha Decay

**기존 논문들**:
- McLean & Pontiff (2016): "Does Academic Research Destroy Stock Return Predictability?"
  - Academic에서 factor 발표 → crowding → returns 하락
  - 우리 contribution: 게임이론 framework로 mechanism 설명

- DeMiguel et al. (2021): "What Alleviates Crowding in Factor Investing?"
  - Crowding 완화 방법론
  - 우리와 다른 각도 (decay mechanism vs mitigation)

- Kang et al. (2021): "Crowding and Factor Returns"
  - Empirical analysis of crowding
  - 우리는 theoretical explanation 추가

- Hua & Sun (2024): "Dynamics of Factor Crowding"
  - Recent work on crowding dynamics
  - 우리의 game-theoretic model과 유사성 확인 필요

**우리의 Novelty**:
✅ **Game-theoretic model**:
- Nash equilibrium에서 alpha decay 도출
- Hyperbolic decay formula: α(t) = K/(1+λt)
- Mathematical rigor로 existing findings 설명

⚠️ **중복 가능성**:
- Hua & Sun (2024)와 비교 필수
- 만약 similar하면 우리 contribution 명확히 구분

**Recommendation**:
- 문헌에서 게임이론 기반 alpha decay 찾기
- 만약 유사하면 차별점 명시
- 만약 novel하면 강조

---

### Topic 2: Domain Adaptation in Finance

**기존 논문들**:
- Long et al. (2015): "Learning Transferable Features"
  - MMD (Maximum Mean Discrepancy) 기반 domain adaptation
  - 우리가 사용한 기본 방법

- Ganin et al. (2016): "Unsupervised Domain Adaptation by Backpropagation"
  - Adversarial domain adaptation (DANN)
  - 우리가 비교한 baseline

- Long et al. (2018): "Conditional Adversarial Domain Adaptation (CDAN)"
  - Conditional adaptation
  - 우리의 regime-conditional idea와 유사

- Financial domain adaptation papers:
  - Sparse (금융에서 DA는 미미)
  - 우리가 처음으로 시도한 영역일 가능성 높음

**우리의 Temporal-MMD (폐기됨)**:
❌ Regime-conditional MMD 시도
- 이론적으로 sound하지만 empirically fails
- Europe에서 -21.5% (심각한 실패)

✅ **실제 novelty**:
- Financial market에 domain adaptation 처음 적용
- Regime 개념 도입 (새로움)
- But: regime transfer 안 됨 (실패)

**Recommendation**:
- Temporal-MMD 제거
- Standard MMD 사용으로 단순화
- "Financial domain adaptation: First application" 강조

---

### Topic 3: Conformal Prediction & Uncertainty

**기존 논문들**:
- Vovk et al. (2005): "Algorithmic Learning in a Random World"
  - 원본 conformal prediction
  - 우리가 foundation으로 사용

- Romano et al. (2019): "Conformalized Quantile Regression"
  - Conformal 기반 quantile regression
  - 우리가 factor returns에 적용

- Angelopoulos & Bates (2021): "Gentle Introduction to Conformal Prediction"
  - 최근 review
  - 우리와 비슷한 문제들 다룸

- Tibshirani et al. (2019): "Conformal Prediction Under Covariate Shift"
  - Covariate shift under conformal
  - Factor crowding = covariate shift 해석 가능

**우리의 Novelty**:
✅ **Application to factor crowding**:
- First application of conformal prediction to crowding
- Novel problem formulation
- Practical uncertainty bounds for practitioners

⚠️ **Potential overlap**:
- Tibshirani et al. (2019)과 covariate shift 해석에서 비슷할 수 있음
- 우리의 crowding-specific interpretation이 differentiation

**Recommendation**:
- 기존 conformal papers와 비교
- Crowding-specific application은 novel
- Covariate shift 관점에서 기존 방법과 구분

---

## 🔍 겹치는 부분 분석

### Topic별 Risk Assessment

| Topic | Paper | Risk | Action |
|-------|-------|------|--------|
| Alpha Decay | JMLR | 🟡 Medium | Hua & Sun (2024) 비교 필수 |
| Domain Adaptation | JMLR/KDD | 🟢 Low | Finance 적용은 new, Temporal-MMD는 폐기 |
| Conformal | ICML | 🟡 Medium | Crowding-specific application이 key novelty |
| Factor Crowding | All | 🟢 Low | Empirical analysis는 new (6 regions) |
| Global Scale | KDD | 🟢 Low | Global scope는 differentiating factor |

### 즉시 조치 필요

**1. Hua & Sun (2024) 완독**
- "Dynamics of Factor Crowding" 세부 검토
- 우리 game-theoretic model과 비교
- 만약 유사하면 우리만의 contribution 명확히

**2. Temporal-MMD 제거**
- ❌ JMLR: Theorem 5 (regime-conditional formula) 제거 vs 유지?
  - **제거 권장**: 이론도 empirically verified 안 됨
  - **대신**: Standard MMD 이론으로 대체
- ❌ KDD: Table 7 결과 완전 제거
  - **대신**: Standard MMD 또는 각 지역별 독립 모델

**3. Conformal Prediction - Covariate Shift 명확화**
- Tibshirani et al.과 우리의 차별점 명시
- Factor crowding = covariate shift의 특정 case임을 강조

---

## 📋 Action Plan

### IMMEDIATE (이번 주)
- [ ] Hua & Sun (2024) 읽기
- [ ] References.bib의 Crowding papers 분석
- [ ] 각 topic별 2-3개 recent papers 더 확인

### THIS WEEK
- [ ] Temporal-MMD 코드 제거
- [ ] Table 7 데이터 재처리 (Standard MMD 결과로)
- [ ] JMLR 논문에서 regime-conditional formula 제거 또는 수정

### NEXT WEEK
- [ ] 각 논문의 novelty statement 업데이트
- [ ] Literature section 재작성
- [ ] Contribution 명확히 구분

---

## 📝 Recommended Paper Changes

### JMLR Paper ("Not All Factors Crowd Equally")

**REMOVE**:
- ❌ Theorem 5 (regime-conditional bound) - 이론은 좋지만 empirically invalid
- ❌ Temporal-MMD method - 실패했으므로 폐기
- ❌ "Regime-conditional matching improves transfer" 주장

**KEEP**:
- ✅ Game-theoretic alpha decay model
- ✅ Conformal prediction for risk
- ✅ Empirical validation on Fama-French factors

**ADD**:
- ✅ Comparison: Our game-theoretic model vs Hua & Sun (2024)
- ✅ Why Temporal-MMD failed: regime non-transfer (pedagogical value)
- ✅ Standard MMD as alternative domain adaptation

**New structure**:
```
1. Introduction
2. Game-Theoretic Model of Alpha Decay ← KEEP
3. Conformal Prediction Framework ← KEEP
4. Global Empirical Validation ← KEEP
5. Literature: Comparison with recent papers ← UPDATE
6. Conclusion
```

### KDD Paper ("Mining Factor Crowding at Global Scale")

**REMOVE**:
- ❌ Temporal-MMD results (Table 7 current version)
- ❌ "Regime-conditional domain adaptation" method
- ❌ Claims about regime transfer

**KEEP**:
- ✅ ML detection methods (LSTM, XGBoost)
- ✅ Global 6-region analysis
- ✅ Factor taxonomy

**ADD**:
- ✅ Standard MMD baseline comparison
- ✅ Why regime transfer fails (case study: Europe vs Japan)
- ✅ Region-specific factors analysis

**New Table 7**:
```
Instead of Temporal-MMD:
- Standard MMD results
- Or: Region-specific independent models
- Or: No transfer attempt (each region separate)
```

### ICML Paper ("Conformal Prediction for Factor Crowding")

**KEEP AS IS** (independent, no changes needed)
- ✅ Conformal prediction framework
- ✅ Coverage guarantees
- ✅ Comparison with Bayesian/Bootstrap

**ADD**:
- ✅ Relate to Tibshirani et al. (2019)
- ✅ Explain why crowding = covariate shift
- ✅ Differentiate from general covariate shift papers

---

## 🎯 Summary of Changes

| 변경사항 | JMLR | KDD | ICML |
|---------|------|-----|------|
| Temporal-MMD 제거 | 제거 | 제거 | N/A |
| Theorem 5 제거 | ❌제거 | N/A | N/A |
| Standard MMD 추가 | ✅ | ✅ | N/A |
| Literature 업데이트 | ✅ | ✅ | ✅ |
| 각 region 독립분석 | N/A | ✅ | N/A |
| Conformal 강화 | ✅ | ✅ | ✅ |

---

## ⚠️ 주의사항

1. **Theorem 5을 제거할지 수정할지**는 아직 미결정
   - 이론은 mathematically correct
   - But empirically invalid (regimes not domain-invariant)
   - 선택: 완전 제거 vs "limited applicability" 주석 추가

2. **Game-theoretic model (Theorem 1)**은 유지
   - Alpha decay formula α(t) = K/(1+λt)
   - Novel하고 empirically 지지됨
   - 다른 논문과 다른 부분

3. **Conformal Prediction**은 강화
   - 가장 강한 부분
   - Covariate shift와 명확히 구분하기

---

## References to Analyze Immediately

**Must read**:
1. Hua & Sun (2024) - "Dynamics of Factor Crowding"
2. Tibshirani et al. (2019) - "Conformal Prediction Under Covariate Shift"
3. DeMiguel et al. (2021) - "What Alleviates Crowding in Factor Investing?"

**Background**:
4. McLean & Pontiff (2016)
5. Long et al. (2018) - CDAN

---

## Next Meeting Checklist

- [ ] Hua & Sun (2024) 분석 결과
- [ ] Theorem 5 결정: 제거 vs 수정 vs 유지
- [ ] Temporal-MMD 완전 제거 실행
- [ ] 각 논문의 novelty statement 최종 버전
- [ ] Literature section 업데이트 계획

**Goal**:
- Remove failed experimental results (Temporal-MMD)
- Clarify novelty vs existing literature
- Strengthen three papers with clear differentiation
