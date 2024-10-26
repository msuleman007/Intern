import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Data loading and preprocessing
df = pd.read_csv("customer_data.csv")
df.fillna(df.mean(), inplace=True)

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

scaler = StandardScaler()
df[['monthly_charges', 'total_charges']] = scaler.fit_transform(df[['monthly_charges', 'total_charges']])

# EDA (visualizations)
sns.countplot(df['churn'])
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Feature Engineering
df['tenure_category'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60], labels=['0-1yr', '1-2yrs', '2-3yrs', '3-4yrs', '4-5yrs'])
df['monthly_tenure_interaction'] = df['monthly_charges'] * df['tenure']

# Model Training
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

roc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC AUC Score: {roc_score:.2f}")

# Save Model
joblib.dump(model, 'churn_model.pkl')
