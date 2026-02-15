"""
CMS Mathematical Modelling Competition 2026
Task 4 – Female Foetus Aneuploidy Classification
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================
# CONFIG
# ============================

FEMALE_FILE = "FemaleFoetus.csv"


# ============================
# LOAD DATA
# ============================

columnNames = [
    'Sample ID', 'Pregnant woman ID', 'Maternal age', 'Maternal height',
    'Maternal weight', 'LMP Date', 'Conception Method', 'Test Date',
    'Blood Draws Count', 'Gestational Age', 'Maternal BMI',
    'Total Raw Reads', 'Aligned Reads Prop', 'Duplicate Reads Prop',
    'Unique Aligned Reads', 'GC Content', 'Z-score 13', 'Z-score 18',
    'Z-score 21', 'Z-score X', 'Z-score Y', 'Y-chrom Conc',
    'X-chrom Conc', 'GC 13', 'GC 18', 'GC 21',
    'Filtered Reads Prop', 'Aneuploidy Detected',
    'Num Pregnancies', 'Num Deliveries', 'Foetal Health Status'
]

data = pd.read_csv(FEMALE_FILE, header=None, names=columnNames)

# Convert numeric columns
numeric_cols = [
    'Z-score 13', 'Z-score 18', 'Z-score 21', 'Z-score X',
    'Maternal BMI', 'GC 13', 'GC 18', 'GC 21',
    'Filtered Reads Prop', 'Aligned Reads Prop'
]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop missing
data = data.dropna(subset=numeric_cols)

# ============================
# CREATE BINARY OUTCOME
# ============================

data['Abnormal'] = data['Aneuploidy Detected'].notna().astype(int)

print("\nClass Distribution:")
print(data['Abnormal'].value_counts())


# ============================
# FEATURE SELECTION
# ============================

features = [
    'Z-score 13', 'Z-score 18', 'Z-score 21', 'Z-score X',
    'Maternal BMI', 'GC 13', 'GC 18', 'GC 21',
    'Filtered Reads Prop', 'Aligned Reads Prop'
]

X = data[features]
y = data['Abnormal']

# Standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# ============================
# LOGISTIC REGRESSION MODEL
# ============================

X_train_sm = sm.add_constant(X_train)
model = sm.Logit(y_train, X_train_sm).fit()

print("\nLogistic Regression Summary:")
print(model.summary())

# ============================
# EVALUATION
# ============================

X_test_sm = sm.add_constant(X_test)
y_pred_prob = model.predict(X_test_sm)
y_pred = (y_pred_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print("\nModel Performance:")
print("Accuracy:", accuracy)
print("AUC:", auc)


# ============================
# ROC CURVE
# ============================

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.show()
