# 📊 Customer Churn Prediction using XGBoost & Power BI

## 📌 Overview
This project predicts **customer churn probability** using **machine learning (XGBoost)** and visualizes the results in **Power BI** for actionable business insights.  
It helps businesses identify **high-risk customers** so that retention strategies can be implemented proactively.

---

## 🛠 Features
- Automated **data preprocessing** using `scikit-learn` Pipelines
- **XGBoost Classifier** for churn prediction
- **Class imbalance handling** using `scale_pos_weight`
- **Hyperparameter tuning** with `RandomizedSearchCV`
- **Model evaluation**: AUC, Accuracy, Confusion Matrix
- **Export to Power BI** for KPI and dashboard creation
- Risk segmentation: **Low**, **Medium**, **High**
- **KPI-ready dataset** with prediction date for trend analysis
- Conditional formatting for churn probability in Power BI

---

## 📂 Dataset
- **Source:** [Kaggle Customer Churn Dataset](https://www.kaggle.com/) *(replace with your dataset link if different)*
- Contains customer demographics, account details, and service usage data
- **Target variable:** `Churn` (1 = customer left, 0 = customer stayed)

---

## ⚙ Tech Stack
- **Python**: Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, SHAP
- **Power BI** for dashboard and KPI visualization
- **Jupyter Notebook** / **Google Colab** for experimentation

---

## 📈 Workflow
1. **Data Loading & Cleaning**
   - Handle missing values with `SimpleImputer`
   - Encode categorical variables using `OneHotEncoder`
   - Scale numerical features using `StandardScaler`

2. **Model Training**
   - XGBoost Classifier with tuned hyperparameters
   - Handle imbalance with `scale_pos_weight`

3. **Evaluation**
   - Calculate AUC, accuracy, and confusion matrix
   - Feature importance analysis using **SHAP values**

4. **Export for Power BI**
   - Create CSV with columns:  
     `CustomerID`, `Actual_Churn`, `Churn_Probability`, `Predicted_Churn`, `Prediction_Date`, `Risk_Level`
   - Sorted by churn probability

5. **Power BI Dashboard**
   - High Risk % KPI card
   - Colored bar chart for churn probability
   - Pie chart for risk level segmentation
   - Trend chart for churn over time

---

## 📊 Example KPIs in Power BI
- **High Churn Risk (%)**
- **Total Customers Analyzed**
- **Predicted Churners Count**
- **Average Churn Probability**
- **Model Accuracy (%)**

---

## 🚀 How to Run

###  Clone Repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

###  Install Dependencies
```bash
pip install -r requirements.txt

### Run the Notebook

# Open customer_churn.ipynb in Jupyter Notebook or Google Colab
# Run all cells to train the model and generate the Power BI CSV

### Load into Power BI
Open Power BI Desktop

# Get Data → CSV → select customer_churn_powerbi_ready.csv
# Create KPIs and visuals as per your dashboard design

📌 Future Improvements
# Improve accuracy using deep learning models (LSTM, ANN)

# Integrate real-time churn alert system

# Add customer lifetime value prediction


