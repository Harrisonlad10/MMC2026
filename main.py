import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ---------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------
# The dataset has no headers, so we define them manually based on Appendix 1
column_names = [
    'Sample ID', 'Pregnant woman ID', 'Maternal age', 'Maternal height', 'Maternal weight',
    'LMP Date', 'Conception Method', 'Test Date', 'Blood Draws Count', 'Gestational Age',
    'Maternal BMI', 'Total Raw Reads', 'Aligned Reads Prop', 'Duplicate Reads Prop',
    'Unique Aligned Reads', 'GC Content', 'Z-score 13', 'Z-score 18', 'Z-score 21',
    'Z-score X', 'Z-score Y', 'Y-chrom Conc', 'X-chrom Conc', 'GC 13', 'GC 18',
    'GC 21', 'Filtered Reads Prop', 'Aneuploidy Detected', 'Num Pregnancies',
    'Num Deliveries', 'Foetal Health Status'
]

# Read the CSV. Note: header=None because the first row is data, not labels.
df = pd.read_csv('Dataset - Male foetuses.csv', header=None, names=column_names)

# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------
# The 'Gestational Age' column is in "11w+6" format (11 weeks + 6 days).
# We need to convert this to a single number (e.g., 11.85 weeks) for math.

# 1. Parse "11w+6" to numbers
def parse_gestational_age(age_str):
    try:
        parts = str(age_str).split('w+')
        weeks = float(parts[0])
        days = float(parts[1])
        return weeks + (days / 7.0)
    except:
        return np.nan

df['Gestational_Age_Weeks'] = df['Gestational Age'].apply(parse_gestational_age)

# 2. FORCE NUMERIC CONVERSION (The Fix)
# This converts any non-numeric junk to NaN (Not a Number) so it doesn't crash the model
cols_to_fix = ['Maternal BMI', 'Y-chrom Conc', 'Maternal age', 'Maternal height', 'Maternal weight']

for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Drop rows with missing values
df_clean = df.dropna(subset=['Maternal BMI', 'Y-chrom Conc', 'Gestational_Age_Weeks'])

# Check if it worked: These should all say 'float64' or 'int64', NOT 'object'
print("Data Types after cleaning:")
print(df_clean[['Gestational_Age_Weeks', 'Maternal BMI', 'Y-chrom Conc']].dtypes)

# ---------------------------------------------------------
# 3. VISUALIZATION (To understand the data)
# ---------------------------------------------------------
# Scatter plot: Gestational Age vs Y-Chromosome Concentration
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_clean, x='Gestational_Age_Weeks', y='Y-chrom Conc', alpha=0.6)
plt.title('Y-Chromosome Concentration vs Gestational Age')
plt.xlabel('Gestational Age (Weeks)')
plt.ylabel('Y-Chromosome Concentration')
plt.grid(True)
plt.show()

# Scatter plot: BMI vs Y-Chromosome Concentration
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_clean, x='Maternal BMI', y='Y-chrom Conc', alpha=0.6)
plt.title('Y-Chromosome Concentration vs Maternal BMI')
plt.xlabel('Maternal BMI')
plt.ylabel('Y-Chromosome Concentration')
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# 4. STATISTICAL CORRELATION
# ---------------------------------------------------------
# We use Spearman correlation because the relationship might not be strictly linear
correlation = df_clean[['Y-chrom Conc', 'Gestational_Age_Weeks', 'Maternal BMI']].corr(method='spearman')
print("\nSpearman Correlation Matrix:")
print(correlation)

# ---------------------------------------------------------
# 5. REGRESSION MODEL (The "Relational Model")
# ---------------------------------------------------------
# We want to predict Y-chrom Conc (Y) based on Age and BMI (X)
# Model: Y_conc = b0 + b1*Age + b2*BMI

X = df_clean[['Gestational_Age_Weeks', 'Maternal BMI']]
y = df_clean['Y-chrom Conc']

# Add a constant (intercept) to the predictors
X = sm.add_constant(X)

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y, X).fit()

# Print the summary stats (p-values, R-squared)
print("\nRegression Model Summary:")
print(model.summary())