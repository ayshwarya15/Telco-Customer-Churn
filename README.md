# 📊 Customer Churn Prediction using Machine Learning

## 🔍 Project Overview
This project predicts customer churn using machine learning. It aims to identify customers likely to leave a telecom company and uncover key factors influencing churn through exploratory data analysis (EDA).

---

## 🎯 Objectives
- Perform EDA to discover patterns related to customer churn.
- Visualize trends across tenure, charges, and services.
- Build a Random Forest Classifier to predict churn.
- Evaluate model performance with precision, recall, and confusion matrix.

---

## 📂 Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- Jupyter Notebook
- Git & GitHub
- Optional: Flask or Streamlit (for deployment)

---

## 📈 Key Insights
- Churn rate is higher among customers with:
  - Month-to-month contracts
  - Fiber optic internet
  - High monthly charges
  - Short tenure
- Lack of tech support and online backup services correlates with higher churn.

---

## 🤖 Model Evaluation

### Classification Report:
| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Not Churned (0) | 0.82      | 0.94   | 0.87     | 1539    |
| Churned (1)     | 0.72      | 0.45   | 0.55     | 574     |
| **Accuracy**    |           |        | **0.80** | 2113    |

- **Weighted Avg F1-Score**: 0.79
- Low recall for churned customers → room for improvement.

---

## 🚀 Future Enhancements
- Use SMOTE for class imbalance
- Try other models like XGBoost or Logistic Regression
- Hyperparameter tuning (GridSearchCV)
- Deploy with Flask or Streamlit
