import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cluster import KMeans
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


#Config
MALE_FILE = "MaleFoetus.csv"
Y_THRESHOLD = 4.0
SIGMA = 0.3  # measurement error assumption
N_CLUSTERS = 4

#Parse age
def parse_gestational_age(age_str):
    try:
        parts = str(age_str).split('w+')
        weeks = float(parts[0])
        days = float(parts[1])
        return weeks + (days / 7.0)
    except:
        return np.nan


#Load data
columnNames = [
    'Sample ID', 'Pregnant woman ID', 'Maternal age', 'Maternal height', 'Maternal weight',
    'LMP Date', 'Conception Method', 'Test Date', 'Blood Draws Count', 'Gestational Age',
    'Maternal BMI', 'Total Raw Reads', 'Aligned Reads Prop', 'Duplicate Reads Prop',
    'Unique Aligned Reads', 'GC Content', 'Z-score 13', 'Z-score 18', 'Z-score 21',
    'Z-score X', 'Z-score Y', 'Y-chrom Conc', 'X-chrom Conc', 'GC 13', 'GC 18',
    'GC 21', 'Filtered Reads Prop', 'Aneuploidy Detected', 'Num Pregnancies',
    'Num Deliveries', 'Foetal Health Status'
]

data = pd.read_csv(MALE_FILE, header=None, names=columnNames)

# Clean numeric columns
numericCols = ['Maternal BMI', 'Y-chrom Conc']
for col in numericCols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data['Gestational_Age_Weeks'] = data['Gestational Age'].apply(parse_gestational_age)

data = data.dropna(subset=['Maternal BMI', 'Y-chrom Conc', 'Gestational_Age_Weeks'])

data['Y-chrom Conc'] = data['Y-chrom Conc'] * 100

#Find earliest threshold time per pregnancy

data['AboveThreshold'] = data['Y-chrom Conc'] >= Y_THRESHOLD

earliest = (
    data[data['AboveThreshold']]
    .groupby('Pregnant woman ID')['Gestational_Age_Weeks']
    .min()
    .reset_index()
    .rename(columns={'Gestational_Age_Weeks': 'Earliest_Threshold_Age'})
)

thresholdData = pd.merge(
    earliest,
    data[['Pregnant woman ID', 'Maternal BMI']],
    on='Pregnant woman ID',
    how='left'
).drop_duplicates()

print("\nTotal male pregnancies reaching threshold:", len(thresholdData))

#BMI Clustering

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
thresholdData['BMI_Group'] = kmeans.fit_predict(
    thresholdData[['Maternal BMI']]
)

#Order groups by mean BMI
groupOrder = thresholdData.groupby('BMI_Group')['Maternal BMI'].mean().sort_values().index
mapping = {old: new for new, old in enumerate(groupOrder)}
thresholdData['BMI_Group'] = thresholdData['BMI_Group'].map(mapping)

#Optimal Timing (95th percentile)
optimalTiming = thresholdData.groupby('BMI_Group')['Earliest_Threshold_Age'].quantile(0.95)

print("\nOptimal NIPT Timing per BMI Group (95% rule):")
print(optimalTiming)

#Regression Model
X = thresholdData[['Maternal BMI']]
X = sm.add_constant(X)
y = thresholdData['Earliest_Threshold_Age']

model = sm.OLS(y, X).fit()

print("\nRegression Model:")
print(model.summary())

#Measurement Error Analysis
def probability_above_threshold(y_value):
    return 1 - norm.cdf(Y_THRESHOLD, loc=y_value, scale=SIGMA)

data['ProbAbove4'] = data['Y-chrom Conc'].apply(probability_above_threshold)

error_analysis = data.groupby('Maternal BMI')['ProbAbove4'].mean()

print("\nMeasurement Error Impact (mean probability by BMI):")
print(error_analysis.head())

#Plotting
plt.figure(figsize=(8,5))
sns.scatterplot(data=thresholdData,
                x='Maternal BMI',
                y='Earliest_Threshold_Age',
                hue='BMI_Group',
                palette='viridis')

plt.title("Earliest Threshold Age vs BMI")
plt.xlabel("Maternal BMI")
plt.ylabel("Earliest Age Reaching Y ≥ 4%")
plt.grid(True)
plt.show()
