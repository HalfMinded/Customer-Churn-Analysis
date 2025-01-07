# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load dataset
file_path = r'C:\Adi\PROJECTS\Churn Rate Analysis\archive\WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(file_path)

# Step 1: Data Cleaning
# Convert 'TotalCharges' to numeric and replace blank spaces with 0
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].replace(' ', '0'))

# Step 2: Exploratory Data Analysis (EDA)
# Churn distribution
churn_distribution = data['Churn'].value_counts(normalize=True)
plt.figure(figsize=(8, 6))
churn_distribution.plot(kind='bar', color=['skyblue', 'salmon'], alpha=0.8)
plt.title('Churn Distribution', fontsize=16)
plt.xlabel('Churn', fontsize=14)
plt.ylabel('Proportion', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Tenure vs Churn
plt.figure(figsize=(10, 6))
data.boxplot(column='tenure', by='Churn', grid=False, patch_artist=True, showmeans=True,
             medianprops=dict(color='red'), meanprops=dict(color='blue'))
plt.title('Tenure Distribution by Churn', fontsize=16)
plt.suptitle('')
plt.xlabel('Churn', fontsize=14)
plt.ylabel('Tenure', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# MonthlyCharges vs Churn
plt.figure(figsize=(10, 6))
data.boxplot(column='MonthlyCharges', by='Churn', grid=False, patch_artist=True, showmeans=True,
             medianprops=dict(color='red'), meanprops=dict(color='blue'))
plt.title('Monthly Charges Distribution by Churn', fontsize=16)
plt.suptitle('')
plt.xlabel('Churn', fontsize=14)
plt.ylabel('Monthly Charges', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Step 3: Feature Encoding
# Encode categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_cols:
    if col != 'customerID':  # Exclude customerID as it's not a feature
        data[col] = le.fit_transform(data[col])

# Step 4: Splitting Data
X = data.drop(['customerID', 'Churn'], axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Development (Random Forest)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

# Step 6: Evaluation
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score and Curve
roc_auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.7)
plt.show()
