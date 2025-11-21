# 📦 Return Prediction Using Machine Learning
## (A Capstone Project for Predictive Modeling & Business Optimization)

## 📘 Project Overview

This capstone project focuses on developing a robust machine learning model to predict whether a given sales transaction is **high-risk for product return**. By leveraging historical transactional, customer, product, and store-level data, the model provides actionable intelligence to operations and supply-chain teams, allowing them to:

* **Reduce Return-Related Losses:** Intervene before the cost is incurred.
* **Improve Inventory Planning:** More accurately forecast post-return stock levels.
* **Enhance Customer Satisfaction:** Proactively address potential issues.

The entire end-to-end workflow is implemented in the notebook.

---

## 🎯 Objective:

To construct a predictive model capable of **accurately identifying high-risk return transactions** so that fulfillment and supply-chain teams can take timely, corrective actions.

## 🔬 Key Tasks & Methodology

The project followed a rigorous, multi-stage data science workflow:

### 1️⃣ Data Preprocessing & Cleaning (The Foundation)

* **Handling missing values** and addressing invalid data entries.
* **Managing extreme outliers** to prevent model distortion.
* **Balancing the imbalanced target variable** (`Is_Return`) using **SMOTE** (Synthetic Minority Over-sampling Technique).

### 2️⃣ Exploratory Data Analysis (EDA) (The Discovery)

* **Behavioral Deep Dive:** Investigating **Customer, Product, and Store return rates**.
* **Correlation Heatmap:** Identifying key relationships for feature selection.
* **Generating Key Insights** to inform the Feature Engineering phase.

### 3️⃣ Machine Learning Modeling (The Core)

* **Models Implemented:** Logistic Regression (Baseline) and Random Forest Classifier (Robustness).
* **Advanced Techniques Used:**
    * **Pipeline creation** using `imblearn` for consistent data transformation.
    * **SMOTE integration** inside the ML pipeline to prevent data leakage.
    * **Threshold Optimization** using **Youden’s J Statistic** (maximizing $TPR - FPR$).

---

## 🧩 Feature Engineering: The 21 Input Variables

Starting with 22 raw columns, 21 highly predictive engineered features were derived and used for the final model training. They are grouped into five distinct categories:

* **🛒 Transaction-Level Features:** **`Unit_Price`**, **`Quantity`**, **`Amount`**, **`Is_Discounted`**.
* **👤 Customer Behavioral Features:** **`Cust_Return_Rate`**, **`Total_Cust_Spent`**, **`Avg_Cust_Spend`**.
* **📦 Product-Level Features:** **`Total_Prod_Txn`**, **`Prod_Return_Rate`**.
* **🏪 Store-Level Features:** **`Total_Store_Txn`**, **`Store_Return_Rate`**.
* **🕒 Time-Based Cyclical Features:** **`Sin_Month`**, **`Cos_Month`**, **`Sin_Weekday`**, **`Cos_Weekday`** (to capture seasonality).

---

## 📊 Model Evaluation: Key Performance Metrics

The success of the models was evaluated using a comprehensive suite of metrics focused on operational utility:

1.  **Precision:** Measures the accuracy of positive predictions (minimizing false alarms).
2.  **Recall (Sensitivity):** Measures the ability to correctly identify all actual return cases (minimizing missed returns).
3.  **F1-Score:** The harmonic mean of Precision and Recall.
4.  **ROC-AUC Score:** Overall measure of model discriminative power.
5.  **Custom Threshold Tuning:** Optimizing the **Precision-Recall Tradeoff** using the **Confusion Matrix** to meet specific business requirements.

---

## 🧰 Technologies & Libraries Used

* **Python** (Core Language)
* **Data Handling & Analysis:** `Pandas`, `NumPy`
* **Machine Learning (Scikit-learn):**
    * **Preprocessing:** `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`
    * **Models:** `Logistic Regression`, `Random Forest`
* **Imbalanced Data:** `Imbalanced-learn` (`imblearn`), **SMOTE**
* **Workflow:** `Pipeline` (integrated with preprocessing + model)
* **Visualization:** `Matplotlib`, `Seaborn`
* **Development Environment:** `Jupyter Notebook`
