import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Load Data
columnNames = [
    'Sample ID', 'Pregnant woman ID', 'Maternal age', 'Maternal height', 'Maternal weight',
    'LMP Date', 'Conception Method', 'Test Date', 'Blood Draws Count', 'Gestational Age',
    'Maternal BMI', 'Total Raw Reads', 'Aligned Reads Prop', 'Duplicate Reads Prop',
    'Unique Aligned Reads', 'GC Content', 'Z-score 13', 'Z-score 18', 'Z-score 21',
    'Z-score X', 'Z-score Y', 'Y-chrom Conc', 'X-chrom Conc', 'GC 13', 'GC 18',
    'GC 21', 'Filtered Reads Prop', 'Aneuploidy Detected', 'Num Pregnancies',
    'Num Deliveries', 'Foetal Health Status'
]

# Read the CSV
data = pd.read_csv('MaleFoetus.csv', header=None, names=columnNames)

#Parse "11w+6" to numbers
def parse_gestational_age(age_str):
    try:
        parts = str(age_str).split('w+')
        weeks = float(parts[0])
        days = float(parts[1])
        return weeks + (days / 7.0)
    except:
        return np.nan

data['Gestational_Age_Weeks'] = data['Gestational Age'].apply(parse_gestational_age)


#Converts non-numeric NaN (Not a Number)
colsToFix = ['Maternal BMI', 'Y-chrom Conc', 'Maternal age', 'Maternal height', 'Maternal weight']

for col in colsToFix:
    data[col] = pd.to_numeric(data[col], errors='coerce')

#Drop rows with missing values
dataClean = data.dropna(subset=['Maternal BMI', 'Y-chrom Conc', 'Gestational_Age_Weeks'])

#Check
print("Data Types after cleaning:")
print(dataClean[['Gestational_Age_Weeks', 'Maternal BMI', 'Y-chrom Conc']].dtypes)


#Scatter plot: Gestational Age vs Y-Chromosome Concentration
plt.figure(figsize=(10, 5))
sns.scatterplot(data=dataClean, x='Gestational_Age_Weeks', y='Y-chrom Conc', alpha=0.6)
plt.title('Y-Chromosome Concentration vs Gestational Age')
plt.xlabel('Gestational Age (Weeks)')
plt.ylabel('Y-Chromosome Concentration')
plt.grid(True)
plt.show()

#Scatter plot: BMI vs Y-Chromosome Concentration
plt.figure(figsize=(10, 5))
sns.scatterplot(data=dataClean, x='Maternal BMI', y='Y-chrom Conc', alpha=0.6)
plt.title('Y-Chromosome Concentration vs Maternal BMI')
plt.xlabel('Maternal BMI')
plt.ylabel('Y-Chromosome Concentration')
plt.grid(True)
plt.show()

#Spearman correlation
correlation = dataClean[['Y-chrom Conc', 'Gestational_Age_Weeks', 'Maternal BMI']].corr(method='spearman')
print("\nSpearman Correlation Matrix:")
print(correlation)

#We want to predict Y-chrom Conc (Y) based on Age and BMI (X)
#Model: Y_conc = b0 + b1*Age + b2*BMI

X = dataClean[['Gestational_Age_Weeks', 'Maternal BMI']]
y = dataClean['Y-chrom Conc']

#Add a constant to the predictors
X = sm.add_constant(X)

#Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y, X).fit()

#Print the summary stats (p values, R squared)
print("\nRegression Model Summary:")
print(model.summary())