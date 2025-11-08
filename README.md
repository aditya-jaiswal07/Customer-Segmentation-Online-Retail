# 🛒 Customer Segmentation — Online Retail (FiveGuys Project)

###  Team Members  
**Aditya Nuli**, **Aditya Jaiswal**, **Abhishek Sharma**, **Amar Anand**, and **Sanskar Garg**

---

## Project Overview

This project demonstrates an **iterative machine learning approach** to customer segmentation and prediction using the **Online Retail dataset**.  
Our key insight: **“Better data beats a better algorithm.”**  
By refining our segmentation strategy from **K-Means** to **Gaussian Mixture Models (GMM)**, we improved model performance by **17 percentage points**, achieving a final accuracy of **94%**.

---

##  Business Objective

In e-commerce, understanding customer behavior is essential for:

1. **Targeted marketing**  
2. **Customer retention**  
3. **Product recommendation**

We aim to:
- **Discover meaningful customer segments** from historical transaction data (unsupervised learning)
- **Build predictive models** that classify new customers into these segments based on early purchase patterns

---

## Dataset

**Source:** [UCI Machine Learning Repository – Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)  
**Period:** December 2010 – December 2011  
**Transactions:** 541,909  

**Features:**
- `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

---

##  Methodology

### 1️⃣ Data Preprocessing
- Handled missing values and duplicate entries  
- Removed cancelled transactions  
- Created **RFM metrics** (Recency, Frequency, Monetary value)

### 2️⃣ Baseline Segmentation — K-Means
- Used elbow and silhouette analysis to find optimal cluster count  
- Generated initial customer segments

### 3️⃣ Improved Segmentation — Gaussian Mixture Model (GMM)
- Replaced K-Means with GMM for soft clustering and better cluster boundaries  
- Achieved more interpretable customer segments

### 4️⃣ Predictive Modeling
- Built **Logistic Regression**, **Random Forest**, and **XGBoost** models  
- Used cluster labels as target variables  
- Evaluated using accuracy, F1-score, and confusion matrix  
- Achieved **94% accuracy** using **Random Forest with GMM labels**

---

##  Results

| Model | Segmentation | Accuracy |
|--------|--------------|----------|
| Logistic Regression | K-Means | 72% |
| Random Forest | K-Means | 77% |
| Random Forest | GMM | **94%** |

**Key takeaway:**  
> Better segmentation (GMM) directly improves predictive accuracy.

---

##  Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Models:** K-Means, GMM, Random Forest, XGBoost  
- **Visualization:** Seaborn, Matplotlib  
- **Notebook:** Jupyter (`FiveGuys_new_updated.ipynb`)

---

##  Insights

- RFM features effectively represent customer purchase behavior  
- GMM captures overlapping customer patterns better than K-Means  
- Improved segmentation leads to higher predictive performance

---



