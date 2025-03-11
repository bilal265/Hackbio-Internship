import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set a consistent plotting style
sns.set_theme(style="whitegrid", palette="muted")

# Load data
url = "https://raw.githubusercontent.com/HackBio-Internship/public_datasets/main/R/nhanes.csv"
df = pd.read_csv(url, sep=',', index_col=0)

# --- Data Cleaning ---
print("Checking and Handling Null Values")
print(df.isnull().sum())

# Fill null values efficiently
df = df.fillna(method='ffill')  # Forward fill for continuity
df = df.fillna({
    'Education': 'High School',
    'MaritalStatus': 'NeverMarried',
    'RelationshipStatus': 'Single',
    'Work': 'None'
}).fillna(0)  # Remaining nulls to 0

print("\nNull Values After Cleaning:")
print(df.isnull().sum())
print("\nSample Data:")
print(df.head(2))

# --- Visualizations ---
# 1. Histograms for BMI, Weight, Weight (lbs), and Age
fig, ax = plt.subplots(2, 2, figsize=(12, 10), tight_layout=True)
sns.histplot(df['BMI'], ax=ax[0, 0], kde=True, color='skyblue', bins=30)
ax[0, 0].set_title('BMI Distribution')
sns.histplot(df['Weight'], ax=ax[0, 1], kde=True, color='salmon', bins=30)
ax[0, 1].set_title('Weight Distribution (kg)')
sns.histplot(df['Weight'] * 2.2, ax=ax[1, 0], kde=True, color='lightgreen', bins=30)
ax[1, 0].set_title('Weight Distribution (lbs)')
sns.histplot(df['Age'], ax=ax[1, 1], kde=True, color='orchid', bins=30)
ax[1, 1].set_title('Age Distribution')
plt.show()

# 2. Scatter Plots for Weight vs Height
# Define improved color palettes
gender_palette = {'male': '#1f77b4', 'female': '#ff7f0e'}  # Blue and Orange (colorblind-friendly)
diabetes_palette = {'No': '#2ca02c', 'Yes': '#d62728'}    # Green and Red
smoking_palette = {'Never': '#9467bd', 'Former': '#8c564b', 'Current': '#e377c2'}  # Purple, Brown, Pink

# Replace 0 in SmokingStatus with 'Never'
df['SmokingStatus'] = df['SmokingStatus'].replace({0: 'Never'})

# Plot with Gender
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Weight', y='Height', hue='Gender', palette=gender_palette, 
                size=10, alpha=0.6, legend='brief')
plt.title('Weight vs Height by Gender')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.legend(title='Gender', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Plot with Diabetes
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Weight', y='Height', hue='Diabetes', palette=diabetes_palette, 
                size=10, alpha=0.6, legend='brief')
plt.title('Weight vs Height by Diabetes')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.legend(title='Diabetes', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Plot with Smoking Status
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Weight', y='Height', hue='SmokingStatus', palette=smoking_palette, 
                size=10, alpha=0.6, legend='brief')
plt.title('Weight vs Height by Smoking Status')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.legend(title='Smoking Status', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# --- Statistical Analysis ---
# Mean Pulse Rate
mean_pulse = df['Pulse'].mean()
print(f"\nMean 60-second Pulse Rate: {mean_pulse:.2f}")

# Diastolic BP Range
bp_range = df['BPDia'].max() - df['BPDia'].min()
print(f"Diastolic BP Range: {bp_range:.2f} (Max: {df['BPDia'].max()}, Min: {df['BPDia'].min()})")

# Income Variance and Std Dev
income_var, income_std = df['Income'].var(), df['Income'].std()
print(f"Income Variance: {income_var:.2f}, Standard Deviation: {income_std:.2f}")

# T-tests (store data once for efficiency)
male_age = df[df['Gender'] == 'male']['Age']
female_age = df[df['Gender'] == 'female']['Age']
diabetes_yes_bmi = df[df['Diabetes'] == 'Yes']['BMI']
diabetes_no_bmi = df[df['Diabetes'] == 'No']['BMI']
committed_alcohol = df[df['RelationshipStatus'] == 'Committed']['AlcoholYear']
single_alcohol = df[df['RelationshipStatus'] == 'Single']['AlcoholYear']

t_age_gender, p_age_gender = stats.ttest_ind(male_age, female_age)
t_bmi_diabetes, p_bmi_diabetes = stats.ttest_ind(diabetes_yes_bmi, diabetes_no_bmi)
t_rs_alcohol, p_rs_alcohol = stats.ttest_ind(committed_alcohol, single_alcohol)

print(f"\nT-test (Age vs Gender): t = {t_age_gender:.2f}, p = {p_age_gender:.3f}")
print(f"T-test (BMI vs Diabetes): t = {t_bmi_diabetes:.2f}, p = {p_bmi_diabetes:.3f}")
print(f"T-test (AlcoholYear没事 vs RelationshipStatus): t = {t_rs_alcohol:.2f}, p = {p_rs_alcohol:.3f}")
