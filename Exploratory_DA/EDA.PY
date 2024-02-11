import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Exploratory_DA\winequality-white.csv', sep=';')

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
df[numerical_features].hist(figsize=(12, 8))
plt.suptitle('Histograms of Numerical Features')
plt.show()

sns.pairplot(df[numerical_features])
plt.suptitle('Pairplot of Numerical Features')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df[numerical_features])
plt.title('Boxplot of Numerical Features')
plt.show()



correlation_matrix = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


df.head()

df.shape

df.info()


summary_stats = df.describe()
print("Summary Statistics:")
print(summary_stats)

skewness = df.skew()
kurtosis = df.kurtosis()
print("\nSkewness:")
print(skewness)
print("\nKurtosis:")
print(kurtosis)


import matplotlib.pyplot as plt

skewness = df.skew()
kurtosis = df.kurtosis()

plt.figure(figsize=(10, 6))
skewness.plot(kind='bar', color='blue')
plt.title('Skewness of Features')
plt.xlabel('Features')
plt.ylabel('Skewness')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
kurtosis.plot(kind='bar', color='red')
plt.title('Kurtosis of Features')
plt.xlabel('Features')
plt.ylabel('Kurtosis')
plt.xticks(rotation=45)
plt.show()


threshold = 7

plt.figure(figsize=(10, 6))
sns.scatterplot(x='residual sugar', y='pH', data=df)
plt.title('Scatter Plot of Features')
plt.xlabel('residual sugar')
plt.ylabel('pH')
plt.show()


cleaned_df = df[(df['residual sugar'] < threshold) & (df['pH'] < threshold)]

cleaned_df['Feature_1_log'] = np.log1p(df['residual sugar'])
cleaned_df['Feature_2_log'] = np.log1p(df['pH'])


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print("Interquartile Range (IQR):")
print(IQR)


z_scores = ((df - df.mean()) / df.std()).abs()
print("\nZ-scores:")
print(z_scores)


threshold = 7


outliers = z_scores > threshold
print("\nOutliers:")
print(outliers)


outliers_count = outliers.sum()
print("\nNumber of outliers in each column:")
print(outliers_count)


import seaborn as sns
import matplotlib.pyplot as plt

z_scores = ((df - df.mean()) / df.std()).abs()


threshold = 7


outliers = z_scores > threshold

plt.figure(figsize=(10, 6))
sns.heatmap(outliers, cmap='coolwarm', cbar=False)
plt.title('Outliers Identified by Z-score')
plt.xlabel('Features')
plt.ylabel('Data Points')
plt.show()



cleaned_df = df[~outliers.any(axis=1)]

print("Original DataFrame shape:", df.shape)
print("Cleaned DataFrame shape:", cleaned_df.shape)





z_scores_cleaned = ((cleaned_df - cleaned_df.mean()) / cleaned_df.std()).abs()


threshold = 7

outliers_cleaned = z_scores_cleaned > threshold

plt.figure(figsize=(10, 6))
sns.heatmap(outliers_cleaned, cmap='coolwarm', cbar=False)
plt.title('Outliers Removed from Cleaned Data')
plt.xlabel('Features')
plt.ylabel('Data Points')
plt.show()


