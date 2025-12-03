# Literature Review: Factor Crowding and Alpha Decay

## 1. The Seminal Paper: McLean & Pontiff (2016)

**"Does Academic Research Destroy Stock Return Predictability?"**
Journal of Finance, 2016

### Key Findings
- Studied 82 characteristics from published academic studies
- **Post-publication decay**: ~50% of alpha disappears after publication
- **Mechanism**: Arbitrage (investors learn from publications) vs. overfitting
- Factors with higher arbitrage costs (idiosyncratic risk) decay less

### Our Use
- Establishes that alpha decay is REAL and measurable
- Provides benchmark: ~50% decay is typical
- Suggests arbitrage (not overfitting) is main driver

Source: [McLean & Pontiff](https://www.fmg.ac.uk/sites/default/files/2020-08/Jeffrey-Pontiff.pdf)

---

## 2. Understanding Alpha Decay (Penasse, 2017)

**Key contribution**: Formalizes alpha decay with a model of investor learning

### Model
- Investors learn about anomalies over time
- Capital flows into discovered anomalies
- Alpha decays as function of AUM

### Empirical Finding
- Sharpe ratio of strategies declines by ~50% after publication
- Decay is faster for low-cost-to-arbitrage factors

Source: [Penasse 2017](https://wp.lancs.ac.uk/fofi2018/files/2018/03/FoFI-2018-0089-Julien-Penasse.pdf)

---

## 3. CFM Paper: "Why and How Systematic Strategies Decay" (2021)

**Industry perspective from Capital Fund Management**

### Key Points
- All systematic strategies decay over time
- Three mechanisms:
  1. Crowding (too much capital)
  2. Regime change (market structure shifts)
  3. Data mining (false discoveries)

### Quantification
- Transaction costs erode ~83% of out-of-sample performance
- Post-publication alpha decay reduces ~57%
- Shorting constraints reduce ~88%

Source: [CFM Paper](https://www.cfm.com/wp-content/uploads/2022/12/312-2021-05-Why-and-how-systematic-strategies-decay.pdf)

---

## 4. Market Microstructure Foundation: Kyle (1985)

**"Continuous Auctions and Insider Trading"**

### Model
- Informed trader vs. noise traders vs. market maker
- Informed trader strategically hides orders in noise
- Price impact: ΔP = λ × Q (Kyle's lambda)

### Relevance to Our Work
- Provides micro-foundation for why capital inflows reduce alpha
- More traders with same info → more price impact → less alpha
- λ is endogenous to number of informed traders

Source: [Kyle Model Notes](https://www.math.cmu.edu/ccf/CCFevents/summer17/abstracts/Notes_June_2017.pdf)

---

## 5. Recent Crowding Research (2023-2024)

### Goldman Sachs (Nov 2023)
- Hedge fund crowding at record high
- 735 funds with $2.4T equity positions
- Momentum factor exposure near record

Source: [GS Research](https://www.gspublishing.com/content/research/en/reports/2023/11/20/c1c0208f-e17d-4152-ac01-22aa849e10dc.html)

### "Where Have All the Alphas Gone?" (CEPR, 2024)
- Meta-analysis of hedge fund performance
- Documents downward trend in reported alphas
- Suggests systematic alpha erosion over time

Source: [CEPR DP18979](https://cepr.org/publications/dp18979)

### Chincarini et al. (2024) - "Crowded Spaces and Anomalies"
- Direct study of crowding effects
- Crowding predicts future underperformance

---

## 6. Game-Theoretic Approaches

### Kyle Model Extensions
- Multiple informed traders: α splits among N informed
- **Key result**: α_i = α_total / N
- As N → ∞, individual alpha → 0

### Equilibrium Stability (Çetin & Larsen, 2023)
- Kyle equilibrium is stable for 1-2 trading periods
- Unstable for 3+ periods → relevant for crowding dynamics

Source: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4521688)

---

## 7. What's Missing (Our Contribution)

### Existing Work
- Documents that alpha decays ✓
- Measures post-publication decline ✓
- Provides microstructure foundation ✓

### Gap We Fill
1. **Explicit N-player model**: Alpha as function of N discoverers
2. **Predictive**: Given crowding proxy, predict remaining alpha
3. **Testable**: Fit model to historical factor returns

### Our Model (Preview)
```
α(t) = K / (1 + λ × N(t))

where:
  K = original alpha capacity
  λ = crowding sensitivity
  N(t) = number of agents trading the factor

Testable prediction:
  d(α)/d(N) < 0 with specific functional form
```

---

## Key Citations for Paper

1. McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? Journal of Finance.

2. Kyle, A. S. (1985). Continuous auctions and insider trading. Econometrica.

3. Penasse, J. (2017). Understanding alpha decay. Working paper.

4. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics.

5. Carhart, M. M. (1997). On persistence in mutual fund performance. Journal of Finance.

6. Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers. Journal of Finance.
