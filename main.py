
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 1. LOAD THE DATA

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
df = pd.read_csv('MaleFoetus.csv', header=None, names=column_names)

# 2. DATA CLEANING
# The 'Gestational Age' column is in "11w+6" format (11 weeks + 6 days).
# We need to convert this to a single number (e.g., 11.85 weeks) for math.

def parse_gestational_age(age_str):
    try:
        # Split "11w+6" into ["11", "6"]
        parts = str(age_str).split('w+')
        weeks = float(parts[0])
        days = float(parts[1])
        # Convert to total weeks
        return weeks + (days / 7.0)
    except:
        return np.nan

# Apply the function to create a new numerical column
df['Gestational_Age_Weeks'] = df['Gestational Age'].apply(parse_gestational_age)

# Drop rows where critical data (BMI, Y-conc, Gestational Age) is missing
df_clean = df.dropna(subset=['Maternal BMI', 'Y-chrom Conc', 'Gestational_Age_Weeks'])

print("Data loaded and cleaned. Shape:", df_clean.shape)
print(df_clean[['Gestational_Age_Weeks', 'Maternal BMI', 'Y-chrom Conc']].head())


# 3. VISUALIZATION (To understand the data)

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


# 4. STATISTICAL CORRELATION

# We use Spearman correlation because the relationship might not be strictly linear
correlation = df_clean[['Y-chrom Conc', 'Gestational_Age_Weeks', 'Maternal BMI']].corr(method='spearman')
print("\nSpearman Correlation Matrix:")
print(correlation)


# 5. REGRESSION MODEL (Fixed)

# Force columns to numeric, turning errors into NaN
cols_to_fix = ['Y-chrom Conc', 'Gestational_Age_Weeks', 'Maternal BMI']
for col in cols_to_fix:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Drop any new NaNs created by the numeric conversion
df_model = df_clean.dropna(subset=cols_to_fix)

# Now define X and y using the strictly numeric df_model
X = df_model[['Gestational_Age_Weeks', 'Maternal BMI']]
y = df_model['Y-chrom Conc']

# Ensure no 'object' types remain
print("Column Types:\n", X.dtypes)

# Add a constant (intercept) to the predictors
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X.astype(float)).fit() # Force float just to be safe
print(model.summary())