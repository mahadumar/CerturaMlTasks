from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseValue'] = california.target

# Show basic info
print(df.head())
print(df.describe())
print(df.info())

#step 2

from sklearn.model_selection import train_test_split

# Separate input features (X) and target (y)
X = df.drop('MedHouseValue', axis=1)
y = df['MedHouseValue']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check shapes of splits
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

#step 3
from sklearn.linear_model import LinearRegression

# Step 1: Create the model
model = LinearRegression()

# Step 2: Train the model on training data
model.fit(X_train, y_train)

# Step 3: Make predictions on test data
y_pred = model.predict(X_test)

# Show some predictions
print("First 5 predictions:", y_pred[:5])
print("Actual values:", y_test[:5].values)

#step 4

from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Calculate RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Scatter plot: Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.plot([0, 5], [0, 5], '--', color='red')  # ideal prediction line
plt.show()
