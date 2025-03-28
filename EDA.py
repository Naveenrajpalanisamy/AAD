import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("NSL_KDD_Train.csv")

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Display first few rows
print("\nFirst 5 rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Summary statistics for numerical features
print("\nSummary Statistics:")
print(data.describe())

# Check class distribution
print("\nClass Distribution:")
print(data['label'].value_counts())

# Visualizing class distribution
plt.figure(figsize=(10,5))
sns.countplot(x='label', data=data, order=data['label'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Class Distribution")
plt.show()

# Checking correlation between numerical features
# Selecting only numerical columns for correlation
numeric_data = data.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12, 6))
sns.heatmap(numeric_data.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()


# Distribution of numerical features
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

data[numerical_columns].hist(figsize=(15,10), bins=30)
plt.suptitle("Distribution of Numerical Features")
plt.show()

# Checking unique values in categorical columns
categorical_columns = ['protocol_type', 'service', 'flag']
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(data[col].unique())
