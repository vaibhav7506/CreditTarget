# Model Assumptions, Limitations & Governance Controls

## 1. Regulatory Context (SR 11-7 / OCC 2011-12)
CreditTarget operates under the supervisory expectations of **Federal Reserve SR 11-7** (Supervisory Guidance on Model Risk Management). This document details key model assumptions, structural limitations, and ongoing monitoring standards.

---

## 2. Core Model Assumptions

### 2.1 Commercial Terms & Loss Given Default (LGD)
- **Constant LGD (0.45):** The platform assumes an unsecured retail credit Loss Given Default of 45%, derived from Basel II/III retail exposure benchmarks. Under severe macroeconomic recessions, LGD may experience upward pressure.
- **Exposure at Default (EAD) Multiplier:** EAD is parameterized by product segment (`BASIC`: 0.75, `PLUS`: 0.70, `PREMIER`: 0.60), reflecting historical credit line drawdown behavior prior to default.

### 2.2 Temporal Stationarity & Lookback Windows
- **Rolling Lookback Windows:** Transaction and support interactions are evaluated over rolling 30-day, 60-day, and 90-day intervals. Events occurring prior to 90 days before snapshot date are excluded from immediate behavioral scoring (historical campaign touches are tracked separately via leak-free cumulative windows).
- **Point-in-Time (PIT) Invariant:** All feature engineering guarantees zero forward-looking information leakage (`event_ts <= snapshot_date`).

---

## 3. Limitations & Fallback Mechanisms

### 3.1 Uncontacted Customers
Customers who have not contacted customer support within 90 days have zero NLP note features. The platform imputes default neutral values (sentiment 0.0, distress 0, complaint severity 0.0), allowing transaction velocity and credit bureau proxies to govern score determination.

### 3.2 Grounded GenAI Validation Reporting
When automated validation memos are synthesized, numeric metrics are programmatically reconciled against verified JSON artifacts (`validation_metrics.json`, `policy_uplift.json`). If any metric fails reconciliation within tolerance ($\pm 0.05$), the system immediately activates deterministic Jinja2 template rendering to ensure zero hallucination in compliance documentation.
