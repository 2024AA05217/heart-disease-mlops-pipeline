
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("notebooks/plots", exist_ok=True)

df = pd.read_csv("data/raw/heart.csv")
df["target"] = (df["target"] > 0).astype(int)

# Basic info
print(df.info())
print(df.describe())

# Missing values
plt.figure()
sns.heatmap(df.isna(), cbar=False)
plt.title("Missing Values Heatmap")
plt.savefig("notebooks/plots/missing_values.png")

# Target balance
plt.figure()
sns.countplot(x="target", data=df)
plt.title("Target Class Distribution")
plt.savefig("notebooks/plots/class_balance.png")

# Histograms
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig("notebooks/plots/feature_histograms.png")

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("notebooks/plots/correlation_heatmap.png")

print("EDA completed. Plots saved in notebooks/plots/")
