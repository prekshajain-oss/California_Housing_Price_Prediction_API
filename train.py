from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import joblib

# Load California housing dataset
data = fetch_california_housing()

x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Total records: {x.shape[0]}")

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize & Train RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    random_state=42
)

print("Training RandomForest model...")
model.fit(x_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R2 Score: {r2:.4f}")
print(f"Average Error: ${mae * 100000:,.0f}")

# Save compressed model (compress=3 keeps file size ~31MB, safely under GitHub 100MB limit)
joblib.dump(model, "house_model.joblib", compress=3)
joblib.dump(list(x.columns), "house_features.joblib")
print("Saved house_model.joblib and house_features.joblib successfully!")
