# Telco-Customer-Churn
Customer Churn Prediction using Machine Learning

**Project Overview**
This project focuses on predicting customer churn for a telecom company using real-world customer data. The goal is to identify patterns and factors leading to customer churn and build a machine learning model that accurately classifies whether a customer is likely to leave the service.

**Objectives**
Perform Exploratory Data Analysis (EDA) to uncover key insights about customer behavior.
Use data visualization to understand churn patterns across services and demographics.
Build a Random Forest Classifier model to predict customer churn.
Evaluate the model using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
Share findings and insights that can help businesses retain customers more effectively.

**Technologies Used**
Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
Jupyter Notebook for analysis and modeling
Git & GitHub for version control and sharing
Optional: Flask or Streamlit for deployment (in progress)

**Key Insights from EDA**
Customers with shorter tenure, higher monthly charges, and fiber internet were more likely to churn.
Churn was also higher among customers with no dependents, month-to-month contracts, and no tech support.

**RESULTS**
Precision for Class 0 (Not Churned): 0.82 → 82% of customers predicted to not churn actually did not churn.
Precision for Class 1 (Churned): 0.72 → 72% of customers predicted to churn actually did churn.
Recall for Class 0 (Not Churned): 0.94 → 94% of actual non-churned customers were correctly predicted as not churned.
Recall for Class 1 (Churned): 0.45 → 45% of actual churned customers were correctly predicted as churned. 
F1-score for Class 0 (Not Churned): 0.87
F1-score for Class 1 (Churned): 0.55
Support for Class 0 (Not Churned): 1539 (larger class)
Support for Class 1 (Churned): 574 (smaller class)
