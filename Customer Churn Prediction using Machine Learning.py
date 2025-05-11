#!/usr/bin/env python
# coding: utf-8

# In[21]:


# Telco Customer Churn - Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')


# In[ ]:





# In[23]:


#  info
print("Info:")
print(df.info())


# In[24]:


# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# In[25]:


print(df.describe())


# In[26]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


# In[27]:


print(df.describe())


# In[28]:


print(df['Churn'].value_counts())
sns.countplot(data=df, x='Churn')


# In[29]:


# Visualizing tenure distribution
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True)
plt.title('Tenure Distribution by Churn')
plt.show()


# In[30]:


# Boxplot for MonthlyCharges by Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn')
plt.show()


# In[31]:


# Heatmap of numerical feature correlations
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[32]:


# Barplot for Contract Type vs Churn
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Contract Type vs Churn')
plt.xticks(rotation=45)
plt.show()


# In[33]:


# Churn rate by Internet Service
internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index')
internet_churn.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Internet Service')
plt.ylabel('Proportion')
plt.show()


# In[35]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Data preprocessing
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})  # Encoding target variable
df = pd.get_dummies(df, drop_first=True)  # One-Hot Encoding for categorical variables

# Splitting the data
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable (Churn)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[ ]:


""""Metrics:
Precision: The proportion of positive predictions (churned customers) that are actually correct.

Precision for Class 0 (Not Churned): 0.82 → 82% of customers predicted to not churn actually did not churn.

Precision for Class 1 (Churned): 0.72 → 72% of customers predicted to churn actually did churn.

Recall: The proportion of actual positives (actual churned customers) that were correctly identified.

Recall for Class 0 (Not Churned): 0.94 → 94% of actual non-churned customers were correctly predicted as not churned.

Recall for Class 1 (Churned): 0.45 → 45% of actual churned customers were correctly predicted as churned. This suggests the model is missing many churned customers (low recall for churn).

F1-Score: The harmonic mean of precision and recall. This metric balances precision and recall, and is important when you have imbalanced classes.

F1-score for Class 0 (Not Churned): 0.87

F1-score for Class 1 (Churned): 0.55

Weighted average F1-score: 0.79 → A reasonable overall performance but indicates room for improvement in predicting churned customers.

Support: The number of actual occurrences of each class in the dataset.

Support for Class 0 (Not Churned): 1539 (larger class)

Support for Class 1 (Churned): 574 (smaller class)"""

