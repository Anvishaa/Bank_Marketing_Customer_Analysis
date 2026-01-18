# Bank Marketing Customer Analytics  
**Understanding who to contact — and why — in outbound marketing campaigns**

---

## 1. Project Overview

This project analyzes historical bank marketing data to understand which customers are more likely to subscribe to a term deposit.

The goal is **not** to build the fanciest machine learning model, but to:
- understand customer behavior,
- explore how prediction confidence affects outreach decisions,
- and reason about real trade-offs between campaign reach and efficiency.

The project is intentionally framed from a **data analyst / business analytics** perspective rather than a pure machine learning one.

---

## 2. Business Context

Banks regularly run outbound marketing campaigns (calls, emails, messages) to promote financial products.  
Each contact has a **cost**, and contacting uninterested customers wastes time and resources.

**Core business question:**  
> Who should we contact so that we capture as many interested customers as possible without wasting outreach effort?

This naturally leads to a **binary decision problem** with uneven costs:
- Missing a potential subscriber → lost revenue
- Contacting an uninterested customer → wasted effort

---

## 3. Dataset

- Source: UCI Machine Learning Repository — Bank Marketing Dataset  
- Size: ~45,000 customer records  
- Target variable: `y`  
  - `1` → customer subscribed  
  - `0` → customer did not subscribe  

### Feature categories
- **Demographic:** age, job, education, marital status  
- **Campaign-related:** contact method, month, previous outcomes  
- **Macroeconomic:** interest rates, employment indicators  

The dataset is **class-imbalanced**, with only ~13% positive outcomes.

---

## 4. Data Preparation

The following steps were taken:
- Converted the target variable to binary
- Treated `"unknown"` values as missing
- Dropped rows with missing values for baseline modeling
- Applied one-hot encoding to categorical variables

The cleaning approach prioritizes **clarity and transparency** over maximum data retention.

---

## 5. Baseline Model

- Model used: **Logistic Regression**
- Rationale:
  - Interpretable
  - Produces probabilities (useful for decision-making)
  - Appropriate for business-facing analysis

Class imbalance was handled using `class_weight="balanced"`.

### Why accuracy was avoided
Because predicting “no” for everyone would already achieve high accuracy.  
Instead, the analysis focused on:
- **Recall** — how many potential subscribers are captured
- **Precision** — how efficient outreach is

---

## 6. Feature Leakage Awareness

One feature, `duration` (call length), showed extremely strong predictive power.

However:
- Call duration is only known **after** a call takes place
- It would not be available when deciding whom to contact

For this reason:
- Models were evaluated **with and without `duration`**
- Results including `duration` were treated as an upper bound, not deployable performance

---

## 7. Feature Ablation & Robustness

To understand whether the model relied too heavily on specific inputs:
- Individual categorical features were removed one at a time
- Overall performance changed very little

**Takeaway:**  
Predictive signal is spread across multiple correlated features rather than driven by a single variable.  
This suggests reasonable robustness instead of fragile dependence on any one attribute.

---

## 8. Single-Feature Models (Building Intuition)

Simple one-feature logistic regression models were trained to understand where signal comes from.

Features explored:
- `duration`
- selected macroeconomic variables

**Observations:**
- `duration` alone performs surprisingly well, reinforcing why it represents post-decision leakage
- Macroeconomic variables have weak but non-zero predictive power on their own

These experiments were done to **build intuition**, not to improve final performance.

---

## 9. Threshold Analysis (Key Insight)

Instead of treating the model as a fixed yes/no classifier, predicted probabilities were analyzed across different decision thresholds.

### Example results (excluding `duration`)

| Threshold | Precision | Recall | Customers Contacted |
|----------|-----------|--------|---------------------|
| 0.20 | 0.14 | 0.96 | ~5300 |
| 0.40 | 0.28 | 0.72 | ~2000 |
| 0.70 | 0.45 | 0.51 | ~880 |

**What this shows:**
- Lower thresholds capture nearly all subscribers but result in many wasted contacts
- Higher thresholds improve efficiency but miss more potential subscribers
- There is no single “best” threshold — the right choice depends on campaign cost and capacity

This reframes the model from **predicting outcomes** to **supporting business decisions**.

---

## 10. Scope and Design Decisions

To keep the project focused and realistic:
- Evaluation emphasized recall–precision trade-offs rather than accuracy
- Model complexity was intentionally kept low to preserve interpretability
- Additional algorithms were not explored, as the goal was understanding behavior and decisions rather than maximizing metrics

---

## 11. Key Analytical Insights

- Some variables (like call duration) explain outcomes well but are unusable for real-time targeting.
- Customer subscription behavior is influenced by multiple overlapping factors rather than any single feature.
- The model is most useful as a **ranking tool**, not a hard decision rule.
- Increasing recall inevitably increases wasted outreach, while higher precision comes at the cost of missed opportunities.
- Macroeconomic indicators provide contextual signal rather than strong standalone predictors.
- Model metrics only make sense when interpreted alongside campaign cost and operational constraints.

---

## 12. Tools Used

- Python
- pandas
- scikit-learn

---

## 13. Closing Note

This project was intentionally kept simple in terms of modeling and heavy on reasoning.  
The emphasis was on understanding trade-offs, questioning assumptions, and knowing when further complexity would not meaningfully improve decisions.

