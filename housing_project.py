from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset for exploration
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Description:")
print(df.describe())
print("\nDataset Info:")
print(df.info())
