# 📊 Validation Instability Analysis  
**Random K-Fold appears stable, but TimeSeriesSplit reveals substantial validation instability through large train–test correlation gaps.**

<img width="612" height="588" alt="Screenshot 2026-01-31 at 10 56 28 PM" src="https://github.com/user-attachments/assets/88579ca0-976f-4956-a01d-2257e78ee023" />

---

**Correlation Stability under Different Data Splitting Strategies**

---

## 🔍 Project Overview

This project investigates **validation instability** caused by different data splitting strategies.

In many machine learning workflows, model performance is commonly evaluated using random cross-validation under the assumption that data are **independently and identically distributed (IID)**.  
However, real-world datasets often contain **implicit temporal structure** and **distributional shifts**, making this assumption invalid.

Instead of focusing on predictive accuracy, this study examines:

> **How stable feature–target relationships remain between training and testing sets under different validation strategies.**

---

## 🎯 Objectives

- Analyze validation stability under different data splitting strategies  
- Compare correlation consistency across folds  
- Investigate the effect of:
  - heavy-tailed target distributions
  - temporal dependency
- Demonstrate why random cross-validation can produce overly optimistic evaluations

---

## 📦 Dataset

**Online News Popularity Dataset**
Kaggle: https://www.kaggle.com/datasets/srikaranelakurthy/online-news-popularity

- Approximately 39,600 news articles  
- 61 numerical features  
- Target variable: number of article shares  

### Dataset characteristics

- Strong **heavy-tailed distribution**
- Presence of a time-related variable: `timedelta`
- Temporal dependency not directly observable from standard EDA

---

## 🧠 Core Idea

Rather than training predictive models, this project evaluates:

> **Whether statistical relationships between features and the target remain consistent between training and testing sets.**

If correlations vary substantially across folds:

- learned patterns may not generalize
- validation results become unreliable
- instability originates from data rather than model choice

---

## 🔬 Methodology

### 1️⃣ Data Splitting Strategies

#### **K-Fold Cross Validation**
- Random shuffling enabled
- Assumes approximate IID sampling
- Commonly used in standard ML pipelines

#### **Time Series Split**
- Preserves temporal ordering
- Training samples strictly precede testing samples
- Violates the IID assumption
- Represents a weak non-IID validation scenario

---

### 2️⃣ Correlation Measures

Two complementary metrics are used:

| Metric | Captures | Characteristics |
|------|------|------|
| Pearson | Linear dependence | Sensitive to outliers |
| Spearman | Monotonic dependence | Robust to heavy-tailed distributions |

---

### 3️⃣ Correlation Gap Definition

For each feature:

Correlation Gap (Δρ) is defined as:
Δρ = ρ_train − ρ_test

We further analyze:

- absolute gap magnitude  
- fold-wise variability  

Large and unstable gaps indicate poor generalization of statistical relationships.

---

## 🔁 Unified Correlation Stability Pipeline

```python
def compute_corr_gap(X, y, splitter):
    """
    Compute train/test Pearson & Spearman correlations per feature per fold,
    and their gaps, using a provided sklearn splitter (e.g., KFold, TimeSeriesSplit).

    'gap' is defined as train minus test correlation.

    Formula: gap = corr_train - corr_test
    """
```
---
## 📊 Results

### Heavy-Tailed Target Distribution
The target variable (number of shares) exhibits a strongly heavy-tailed distribution, with extreme values far from the median. Such distributions are known to amplify statistical instability and sensitivity to data splitting strategies.

<img width="609" height="437" alt="Heavytailhisto" src="https://github.com/user-attachments/assets/746da4c1-ce9d-4eea-99dd-2e65ae14f05a" />
<img width="651" height="453" alt="qqplot" src="https://github.com/user-attachments/assets/7c1a8fcd-4a3e-4d96-b471-3625bfa5ba98" />

---

### K-Fold Cross Validation
Random splitting shows stable correlation structures across folds.

<img width="822" height="391" alt="kfold pearson" src="https://github.com/user-attachments/assets/495bf28a-14cf-4d2d-aade-d4bec10d22bb" />
<img width="816" height="394" alt="kfold spearman" src="https://github.com/user-attachments/assets/2f65c999-dc4b-43a2-9e55-1dc0ae5d29bd" />

---

### Time Series Split
Temporal splitting exposes significant instability and fold-wise variance.

<img width="818" height="386" alt="timepearson" src="https://github.com/user-attachments/assets/12d8723c-458f-4f37-85eb-a80186da7352" />
<img width="819" height="396" alt="timespearman" src="https://github.com/user-attachments/assets/190685ce-33dc-4564-a08f-42683a5c07d8" />

---

### Correlation Gap Summary (per fold)

Random K-Fold Split

<img width="529" height="112" alt="Screenshot 2026-01-31 at 10 32 30 PM" src="https://github.com/user-attachments/assets/144e81c3-47e6-48c7-9813-d5da4962526d" />

Time-Series Split

<img width="522" height="111" alt="Screenshot 2026-01-31 at 10 37 03 PM" src="https://github.com/user-attachments/assets/6abe5d2c-9523-489b-8141-ce24c3e276f6" />

**Key observation**
- Time-series split exhibits consistently larger correlation gap magnitude and higher fold-to-fold variability compared to random K-Fold splitting.
- This indicates instability of feature–target relationships under temporal distribution shift.

---

### Comparison

Across all experiments, random K-Fold splitting exhibits consistently small correlation gaps with low fold-wise variance, indicating stable feature–target relationships under approximate IID sampling.

In contrast, time-series splitting results in substantially larger correlation gaps and pronounced dispersion across folds. This instability suggests that
feature–target relationships are not stationary over time, and that random cross-validation masks underlying distribution shifts.

Overall, these results demonstrate that validation stability is highly sensitive to the chosen data splitting strategy, even when identical features and metrics
are used.

## 💡 Discussion

Although standard exploratory data analysis suggests that the dataset is approximately homogeneous, temporal dependency is primarily an ordering-based
property and may remain hidden under random shuffling.

Our results indicate that validation instability arises not from model misspecification, but from distributional shifts over time. Consequently,
random cross-validation can produce overly optimistic evaluation results when temporal structure is implicitly present.

These findings highlight the importance of aligning validation strategy withthe data-generating process, rather than relying solely on model complexity or performance metrics.












