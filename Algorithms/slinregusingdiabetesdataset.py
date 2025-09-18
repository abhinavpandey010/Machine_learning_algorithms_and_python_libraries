import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# Load the real diabetes dataset
diabetes = load_diabetes()

print(diabetes.feature_names)#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

X = diabetes.data # input features
y = diabetes.target # target values (disease progression)

# Split the dataset into training/testing sets
# With random_state (reproducible)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

# Without random_state (random split every time)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)

# Create a linear regression model

model1 = LinearRegression()
model2 = LinearRegression()

# Train both models
model1.fit(X_train1, y_train1)
model2.fit(X_train2, y_train2)

# Predict on test sets
y_pred1 = model1.predict(X_test1)
y_pred2 = model2.predict(X_test2)

# Evaluate both models
# Print actual and predicted values side by side
results = pd.DataFrame({
"Actual": y_test1,
"Predicted": y_pred1
})
print(results)

print("Model with random_state=42:")
print(" R2 Score:", r2_score(y_test1, y_pred1))
print(" MSE:", mean_squared_error(y_test1, y_pred1))

print("Model WITHOUT random_state:")
print(" R2 Score:", r2_score(y_test2, y_pred2))
print(" MSE:", mean_squared_error(y_test2, y_pred2))

# -----------------------------------------
# Optional: Visualize predictions for one feature (e.g., BMI)
plt.figure(figsize=(10, 5))
plt.scatter(X_test1[:, 2], y_test1, color="blue", label="Actual") # BMI is at index 2
plt.scatter(X_test1[:, 2], y_pred1, color="red", label="Predicted", alpha=0.6)
plt.title("Linear Regression: Actual vs Predicted (BMI feature)")
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.legend()
plt.grid(True)
plt.show()