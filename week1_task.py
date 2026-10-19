# Week 1 - Data Cleaning and Exploratory Data Analysis
# I am using the diabetes dataset that comes with scikit-learn, so there is
# no need to download a separate file.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes

output_folder = Path("week1_outputs")
output_folder.mkdir(exist_ok=True)

diabetes = load_diabetes(as_frame=True)
data = diabetes.frame.copy()

print("Number of rows and columns:", data.shape)
print("\nColumn data types:")
print(data.dtypes)

print("\nMissing values in each column:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())

# Remove duplicate rows if any are present.
data = data.drop_duplicates()
data.to_csv(output_folder / "cleaned_diabetes.csv", index=False)

print("\nSummary of the cleaned data:")
print(data.describe().round(2))

# 1. Check how the target values are distributed.
plt.figure(figsize=(7, 4))
sns.histplot(data["target"], bins=25, kde=True)
plt.title("Distribution of Diabetes Target Values")
plt.xlabel("Target value")
plt.tight_layout()
plt.savefig(output_folder / "target_distribution.png", dpi=150)
plt.close()

# 2. Compare the BMI feature with the target value.
plt.figure(figsize=(7, 4))
sns.scatterplot(data=data, x="bmi", y="target", alpha=0.7)
plt.title("BMI Feature and Target")
plt.tight_layout()
plt.savefig(output_folder / "bmi_target_scatter.png", dpi=150)
plt.close()

# 3. Look at correlations between the numeric columns.
plt.figure(figsize=(9, 7))
sns.heatmap(data.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Correlation Between Numeric Columns")
plt.tight_layout()
plt.savefig(output_folder / "correlation_heatmap.png", dpi=150)
plt.close()

print("\nFinished. The cleaned data and charts are in", output_folder)
