# Model Risk Validation Memo: CreditTarget Engine
**Compliance Framework:** Federal Reserve SR 11-7 / OCC 2011-12
**Executive Verdict:** PASS (8/8 rules passed)
**Timestamp:** 2026-11-25T07:01:16.572785+00:00

---

## 1. Executive Summary & Lineage Audit
- **Git Commit SHA:** `unknown`
- **Gold Delta Lake Version:** `v4`
- **Champion Propensity Model Version:** `v5`
- **Champion PD Model Version:** `v5`
- **Storage Backend:** `local`

---

## 2. Model Performance & Stability Metrics

### 2.1 Out-of-Time (OOT) Discriminatory Power
| Metric | Propensity Model (OOT) | Probability of Default Model (OOT) | Governance Gate | Status |
| :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.9102 | 0.8542 | N/A | Tracked |
| **Gini (2*AUC - 1)** | 0.8205 | 0.7084 | >= 0.35 / >= 0.40 | PASS |
| **KS Statistic** | 71.51% | 56.89% | Max Drop <= 15.0% | PASS |
| **Brier Score** | 0.0913 | 0.122 | <= 0.20 | PASS |

### 2.2 Population Stability Index (PSI) & CSI Drift
- **Propensity Score PSI (Val -> OOT):** 0.0162 (Threshold: <= 0.25)
- **PD Score PSI (Val -> OOT):** 0.0018 (Threshold: <= 0.25)
- **Top Feature CSI:**

  - `credit_limit`: 0.0

  - `max_dpd_90d`: 0.0

  - `mean_util_90d`: 0.0004

  - `spend_90d`: 0.0009


---

## 3. Decile Lift Table (OOT Evaluation)
| Decile | Count | Responders | Response Rate | Lift | Cum. Capture | Mean Propensity | Avg PD | Cum. EV ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| 1 | 600 | 543 | 0.905 | 3.65x | 0.3649 | 0.9194 | 0.1787 | $427492.33 |

| 2 | 600 | 453 | 0.755 | 3.04x | 0.6694 | 0.7634 | 0.2287 | $633540.25 |

| 3 | 600 | 239 | 0.3983 | 1.61x | 0.83 | 0.2205 | 0.2514 | $686368.67 |

| 4 | 600 | 91 | 0.1517 | 0.61x | 0.8911 | 0.0966 | 0.2257 | $709960.35 |

| 5 | 600 | 52 | 0.0867 | 0.35x | 0.9261 | 0.0572 | 0.2203 | $715821.24 |

| 6 | 600 | 27 | 0.045 | 0.18x | 0.9442 | 0.0356 | 0.2637 | $710121.24 |

| 7 | 600 | 23 | 0.0383 | 0.15x | 0.9597 | 0.0178 | 0.2622 | $705842.81 |

| 8 | 600 | 26 | 0.0433 | 0.17x | 0.9772 | 0.0141 | 0.2657 | $698651.6 |

| 9 | 600 | 24 | 0.04 | 0.16x | 0.9933 | 0.0112 | 0.2327 | $691277.66 |

| 10 | 600 | 10 | 0.0167 | 0.07x | 1.0 | 0.0076 | 0.246 | $683455.75 |


- **Top-2 Decile Lift:** 3.35x (Minimum Required: 1.80x)

---

## 4. Business Impact: Risk-Gated EV Targeting vs. Naive Baseline
Comparing targeting across identical 30% portfolio budget capacity (2400 customers):

- **Risk-Gated Total EV:** $1454482.52
- **Naive Propensity-Only Total EV:** $940409.38
- **Net EV Uplift:** **+$514073.13** (+54.66%)
- **Avoided Expected Credit Losses:** **$1628090.97**
- **Suppression Breakdown:**

  - `ACTIVE_DELINQUENCY`: 640 customers

  - `BUDGET_CUTOFF`: 382 customers

  - `NEGATIVE_EV`: 837 customers

  - `NONE`: 2400 customers

  - `PD_CAP_EXCEEDED`: 3741 customers


---

## 5. Demographic Fairness & Disparate Impact
- **Overall Minimum Disparate Impact Ratio:** 0.8846 (Benchmark: >= 0.80)
- **Region Code Breakdown:** Min Ratio = 0.8846
- **Age Band Breakdown:** Min Ratio = 0.9342

---

## 6. Model Assumptions & Limitations
1. **Stationarity of Macro Factors:** LGD is fixed at 0.45; extreme macroeconomic downturns may require dynamic LGD stress adjustments.
2. **Support Note Coverage:** NLP distress scoring applies only to customers contacting support within 90 days; non-contacting customers rely solely on transaction and credit history.
3. **Execution Horizon:** 30-day acceptance and 12-month default windows assume constant marketing yield structures over the targeting cycle.