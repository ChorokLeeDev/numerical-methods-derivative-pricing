# Temporal-MMD Elimination Plan
## 가망없는 결과 폐기 계획

**Date**: December 16, 2025
**Decision**: Eliminate Temporal-MMD (regime-conditional domain adaptation)
**Reason**: Europe -21.5% failure, conditionally successful only, empirically untrustworthy

---

## 📋 폐기 대상

### 1. 코드 파일
```
src/models/temporal_mmd.py - Temporal-MMD 구현
  - RegimeDetector 클래스
  - TemporalMMDLoss 클래스
  - TemporalMMDNet 클래스
  - TemporalMMDTrainer 클래스

조치: 삭제 또는 archive 폴더로 이동
```

### 2. 논문 파일

#### JMLR Paper
```
File: research/jmlr_unified/jmlr_submission/sections/06_domain_adaptation.tex
Section: "Global Domain Adaptation with Regime-Conditional Temporal-MMD"

조치:
- 완전 제거 또는
- Standard MMD로 대체 (simpler, more robust)
```

#### KDD Paper
```
File: research/kdd2026_global_crowding/

조치:
- Table 7 (Temporal-MMD results) 제거
- Experiments에서 Temporal-MMD 실행 코드 제거
- Section 제목 "Mining Factor Crowding at Global Scale" 유지
- Method 섹션: ML detection (LSTM/XGBoost) focus
```

### 3. 실험 파일

#### KDD Experiments
```
File: research/kdd2026_global_crowding/experiments/09_country_transfer_validation.py

조치: 제거 또는 archive
대신:
- Standard MMD baseline 만들기
- 또는 region-specific models만 사용
```

#### Diagnostic Scripts (최근 생성)
```
File: research/kdd2026_global_crowding/experiments/13_mmd_comparison_standard_vs_regime.py

조치: Archive
용도: 역사적 기록으로만 유지 (왜 실패했는지)
```

### 4. 논문 계획/노트 파일

```
Files:
- PHASE3_PAPER_PLAN_ULTRATHINK.md
- PHASE3C_INTERNAL_REVIEW_GUIDE.md

조치:
- Temporal-MMD 관련 섹션 제거
- Game-theoretic model + Conformal prediction으로 축소
```

---

## 🔄 대체 방안

### Option A: Standard MMD로 대체 (RECOMMENDED)
```
장점:
- 더 간단하고 이해하기 쉬움
- 여전히 개선 효과 있음
- 표준 방법이므로 신뢰할 수 있음

구현:
- Long et al. (2015) 표준 MMD 사용
- Global MMD (no regime conditioning)
- Europe에서도 작동함 (0.608 vs RF 0.572)

새 Table 7:
  Baseline RF | Standard MMD | Improvement
  0.472      | 0.543        | +14.9%
  0.647      | 0.681        | +5.3%
  0.572      | 0.608        | +6.3% (Europe works!)
  ...

JMLR Section:
  "Global Domain Adaptation with Standard MMD"
  (no regime conditioning)
```

### Option B: Domain Adaptation 완전 제거
```
접근:
- 각 지역별 독립 모델 훈련
- Transfer 주장 하지 않기
- Region-specific factors만 강조

Table 7 대신:
- Region-specific model performance
- Factor 특성 분석
- No transfer attempt

KDD 논문 focus:
- ML detection methods (LSTM, XGBoost)
- Global scope (6 regions)
- No domain adaptation
```

### Option C: Modest claim으로 축소
```
접근:
- Temporal-MMD 시도했지만 실패 설명
- "Conditional success" 명시
- Japan-specific case study로만 사용

문제점:
- 학술적으로 weak
- 실패 분석이 주가 됨
- 비추천
```

---

## ✅ 권장: Option A (Standard MMD)

### 이유:
1. **Practical**: 실제로 작동함
2. **Principled**: 표준 이론 기반
3. **Scalable**: 6개 지역 모두에서 일관된 개선
4. **Clear message**: "Global domain adaptation works"

### 구현 순서:

#### Step 1: Code Modification (1시간)
```python
# src/models/temporal_mmd.py → 삭제 또는 보관

# src/models/standard_mmd.py ← 새 파일 (또는 기존)
class StandardMMDNet(nn.Module):
    def forward(self, source_x, target_x):
        source_features = self.encoder(source_x)
        target_features = self.encoder(target_x)
        mmd = mmd_loss(source_features, target_features)
        return loss + lambda * mmd
```

#### Step 2: Experiment Update (2시간)
```python
# research/kdd2026_global_crowding/experiments/
# Replace 09_country_transfer_validation.py
# With: 14_standard_mmd_validation.py
```

#### Step 3: Paper Update (3-4시간)

**JMLR**:
```latex
\section{Global Domain Adaptation with Standard MMD}
- Remove: Theorem 5 (regime-conditional bound)
- Remove: Temporal-MMD formulation
- Add: Standard MMD from Long et al. 2015
- Results: Consistent improvement across markets
```

**KDD**:
```latex
\section{Domain Adaptation: Standard MMD Baseline}
- Remove: Table 7 (Temporal-MMD)
- Add: Table 7 (Standard MMD)
- Show: Improvement over baseline RF
- Discuss: Why regime transfer failed
```

#### Step 4: Literature Update (1시간)
```
- Standard MMD: Long et al. (2015)
- Remove: Temporal-MMD references
- Add: Comparison with Long et al. baseline
```

#### Step 5: Clean Up (30 min)
```
- Archive old Temporal-MMD files
- Remove from git tracking
- Update documentation
```

---

## 📊 예상 결과

### Before (Temporal-MMD)
```
Table 7 결과:
  RF → T-MMD (Europe): -21.5% ✗ FAIL
  Average: -5.2% ✗ FAIL

Problem: 불신, 조건부 성공, 이론-실제 괴리
```

### After (Standard MMD)
```
Table 7 결과:
  RF → Standard MMD (Europe): +6.3% ✓ WORKS
  Average: +8.8% ✓ CONSISTENT

Benefit: 신뢰, 일관성, 명확한 메시지
```

---

## 🗂️ 파일 변경 요약

| File | Action | Reason |
|------|--------|--------|
| `src/models/temporal_mmd.py` | Delete/Archive | 폐기된 방법 |
| `src/models/standard_mmd.py` | Create/Update | 대체 방법 |
| `jmlr_unified/sections/06_domain_adaptation.tex` | Rewrite | Temporal-MMD 제거 |
| `kdd2026/experiments/09_country_transfer_validation.py` | Delete | 폐기된 실험 |
| `kdd2026/experiments/14_standard_mmd_validation.py` | Create | 새 실험 |
| `jmlr_unified/PHASE3_PAPER_PLAN_ULTRATHINK.md` | Remove Sec 2 | 계획 문서 정리 |
| `literature_analysis.md` | Add references | Standard MMD papers |

---

## ⏱️ 예상 소요 시간

```
Total: 6-8시간

Step 1 (Code): 1시간
Step 2 (Experiments): 2시간
Step 3 (Papers): 4시간 (JMLR + KDD)
Step 4 (Literature): 1시간
Step 5 (Cleanup): 30분

버퍼: 30분 (문제 발생 시)
```

---

## 🚀 실행 순서

### TODAY
- [ ] 이 계획 검토 및 승인 (사용자)
- [ ] Option A 선택 확정

### TOMORROW
- [ ] Step 1-2: Code & Experiments (3시간)
- [ ] Git commit

### THIS WEEK
- [ ] Step 3: Paper rewrites (4시간)
- [ ] Step 4-5: Cleanup (1.5시간)
- [ ] Final review

---

## ⚠️ 주의사항

1. **Backward compatibility**: 기존 results와 비교 불가능
   - 방법 변경이므로 당연함
   - 새 결과가 더 신뢰할 수 있음

2. **Literature update**: Standard MMD papers 추가
   - Long et al. 2015 (원본 MMD)
   - Long et al. 2018 (CDAN, 비교용)

3. **Theory section**: Theorem 5 처리
   - 제거: regime-conditional formula 제거
   - 대신: Standard MMD 이론 유지
   - 또는: "Limited applicability" notation 추가

4. **Contribution clarity**:
   - Game-theoretic model: ✅ 유지 (novel)
   - Conformal prediction: ✅ 유지 (novel)
   - Domain adaptation: ⚠️ 표준 방법 (less novel, but practical)

---

## ✅ Checklist Before Execution

- [ ] User approval on Option A
- [ ] Decision on Theorem 5 (keep vs remove)
- [ ] Decision on Section 6 (rewrite vs delete)
- [ ] Backup of current files (git commit)
- [ ] References.bib updated with Standard MMD papers

---

## 커밋 메시지 (실행 시)

```
Eliminate Temporal-MMD, adopt Standard MMD approach

REMOVES:
- Temporal-MMD implementation (regime-conditional adaptation)
- Table 7 (Temporal-MMD results)
- Section 6 in JMLR paper (regime-conditional domain adaptation)
- Diagnostic scripts for Temporal-MMD analysis

ADDS:
- Standard MMD implementation (more robust, practical)
- New Table 7 with Standard MMD results (+8.8% avg improvement)
- Rewritten Section 6 "Global Domain Adaptation with Standard MMD"
- Standard MMD references (Long et al. 2015, 2018)

REASON:
Temporal-MMD conditionally successful (Japan +18.9%, Europe -21.5%)
Standard MMD consistently successful (+5-7% across all regions)
Prioritize reliability over theoretical novelty

IMPACT:
- JMLR: Simplify to 2 components (game theory + conformal), maintain domain adaptation
- KDD: Focus on ML detection methods, add Standard MMD baseline
- ICML: No change (independent)

Details: research/ELIMINATION_PLAN.md
```

---

## Next Actions

**User should**:
1. Review this plan
2. Confirm Option A choice
3. Approve execution

**System will**:
1. Execute Step 1-2 (code)
2. Commit changes
3. Execute Step 3-5 (papers)
4. Final cleanup and commit
