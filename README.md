# 🛒 Customer Segmentation — Online Retail (FiveGuys Project)

---

## Project Overview

This project implements an end-to-end pipeline that turns transactional data into actionable customer segments and a production-ready classifier for online retail.  
We compute RFM features, use **Gaussian Mixture Models (GMM)** to discover three behavioral segments, and train an optimized **XGBoost** classifier on the GMM labels. The final XGBoost model achieves **96.1% test accuracy**, demonstrating that improved segmentation significantly raises downstream predictive performance. 

---

## Business Objective

In e-commerce, understanding customer behavior enables:
1. **Targeted marketing**  
2. **Customer retention**  
3. **Product / campaign personalization**

Goals:
- Discover meaningful customer segments from historical transactions (unsupervised).  
- Build a predictive model that classifies new/early customers into these segments for targeted actions.

---

## Dataset

**Source:** UCI Online Retail Dataset (Dec 2010 — Dec 2011)  
**Transactions:** 541,909 (raw)  
**Key columns:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

**Working sample (model experiments):** training ≈ 3,470, test ≈ 868 (empirical split used in experiments). 

---

## Methodology

1. **Data preprocessing**
   - Clean transactions (drop duplicates, handle missing CustomerID), remove cancelled invoices, and compute standard **RFM** metrics and derived behavioral features.
2. **Baseline segmentation — K-Means**
   - Used elbow/silhouette analysis to get a baseline cluster solution (K-Means baseline accuracy reported in slides).
3. **Improved segmentation — Gaussian Mixture Model (GMM)**
   - GMM produced three soft clusters: **Champions (≈5.7%)**, **Potential Loyalists (≈38.7%)**, **Occasional Buyers (≈55.5%)**. These labels were used as supervised targets. 
4. **Predictive modeling**
   - Trained and compared Logistic Regression, Random Forest, and **XGBoost** on features independent of the clustering variables.  
   - Feature vector size used for modeling: **m = 16** (behavioral metrics like frequency_trend, avg_days_between_orders). 
   - XGBoost hyperparameters (example): `max_depth=3`, `learning_rate=0.05` (tuned via CV). 
5. **Scalability — Coreset via Sensitivity Sampling**
   - Constructed coresets to speed training while preserving accuracy. The 30% coreset retained **~95.2%** test accuracy with **~1.6×** speedup; 10% coreset gave **~92%** accuracy with **~2.02×** speedup. This demonstrates a practical training/accuracy trade-off for large-scale deployment. 

---

## Results

| Model        | Segmentation | Test Accuracy |
|--------------|--------------|---------------|
| Logistic Reg | K-Means      | (baseline)    |
| Random Forest| K-Means      | (baseline)    |
| **XGBoost**  | **GMM**      | **96.1%**     |

**Notes:** XGBoost trained on GMM labels produced the best generalization (test ≈ **96.1%**, train ≈ 98.13%) with low training time (~0.30s on the training set). GMM→XGBoost substantially outperformed the K-Means baseline (K-Means baseline ≈ 74.9%). 

**Key takeaway:** Better segmentation (soft clusters from GMM) directly improves the supervised classifier’s accuracy.

---

## Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn  
- **Models / Techniques:** RFM feature engineering, K-Means, GMM, Random Forest, XGBoost, coreset (sensitivity sampling)  
- **Notebook:** `customer_segmentation-2.ipynb` 

---

## Insights & Future Work

- RFM and derived behavioral metrics strongly separate customer behaviors.  
- GMM’s soft clustering captures overlapping behavior better than hard K-Means, improving label quality for the supervised step.
- Coreset sampling is a practical speed/accuracy trade-off for large datasets; future work includes applying the coreset pipeline to N > 1M rows and recursive micro-segmentation for the large “Occasional Buyers” cohort. 
