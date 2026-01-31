# 📊 Validation Instability Analysis  
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

\[
\text{Correlation Gap} = \rho_{\text{train}} - \rho_{\text{test}}
\]

We further analyze:

- absolute gap magnitude  
- fold-wise variability  

Large and unstable gaps indicate poor generalization of statistical relationships.

---

## 🔁 Unified Correlation Stability Pipeline

```python
def compute_corr_gap(X, y, splitter):
    """
    Compute train/test Pearson & Spearman correlations per feature per fold.
    Gap = correlation_train − correlation_test
    """
