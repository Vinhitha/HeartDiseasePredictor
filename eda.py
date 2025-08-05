import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heart.csv")
print(df.head())   # see first 5 rows
print(df.info())   # see columns & data types
print(df.describe())  # summary stats

# Count of target classes (0 = No Disease, 1 = Disease)
sns.countplot(x='target', data=df)
plt.title("Heart Disease vs No Disease")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
