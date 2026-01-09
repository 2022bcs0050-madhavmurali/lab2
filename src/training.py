import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


STUDENT_NAME = "Madhav Murali"  # Replace with your actual name
ROLL_NUMBER = "2022BCS0050" # Replace with your actual roll number

# Experiment settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_TYPE = "LinearRegression"  # Options: "LinearRegression", "Lasso", "Ridge"
ALPHA = 0.1 # Only used for Lasso/Ridge

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
# Ensure data folder exists or adjust path as necessary
df = pd.read_csv('../data/winequality-red.csv', sep=';')

X = df.drop('quality', axis=1)
y = df['quality']

# ---------------------------------------------------------
# 2. Pre-processing & Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------
if MODEL_TYPE == "LinearRegression":
    model = LinearRegression()
elif MODEL_TYPE == "Lasso":
    model = Lasso(alpha=ALPHA)
elif MODEL_TYPE == "Ridge":
    model = Ridge(alpha=ALPHA)
else:
    raise ValueError("Unknown model type")

model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model: {MODEL_TYPE}")
print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")

# ---------------------------------------------------------
# 5. Save Artifacts (Model & Metrics)
# ---------------------------------------------------------
os.makedirs('../models', exist_ok=True)

# Save Model
joblib.dump(model, '../models/model.pkl')

# Save Metrics to JSON
metrics = {
    "model_type": MODEL_TYPE,
    "mse": mse,
    "r2_score": r2,
    "hyperparameters": {
        "test_size": TEST_SIZE,
        "alpha": ALPHA if MODEL_TYPE in ["Lasso", "Ridge"] else "N/A"
    }
}

with open('../models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# ---------------------------------------------------------
# 6. Generate GitHub Step Summary
# ---------------------------------------------------------

summary_content = f"""
# 🍷 Wine Quality Prediction Experiment

**Student:** {STUDENT_NAME} ({ROLL_NUMBER})

## 📊 Experiment Results
| Metric | Value |
| :--- | :--- |
| **Model Type** | {MODEL_TYPE} |
| **MSE** | {mse:.4f} |
| **R² Score** | {r2:.4f} |

## ⚙️ Configuration
- **Test Split:** {TEST_SIZE}
- **Alpha:** {ALPHA}
"""

with open('../models/summary_report.md', 'w') as f:
    f.write(summary_content)