import pandas as pd

# Load Titanic dataset from seaborn
import seaborn as sns
df = sns.load_dataset('titanic')

# Show columns
print(df.columns)
print(df.head())

#step 2
# Select features
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df = df[features + ['survived']].dropna()

# Encode categorical features
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Define X and y
X = df[features]
y = df['survived']

print("Processed dataset shape:", X.shape)

from sklearn.ensemble import RandomForestClassifier

#step 3
# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

#step 4
import matplotlib.pyplot as plt

# Get importance scores
importances = model.feature_importances_
feature_names = X.columns

# Plot
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances (Titanic Dataset)")
plt.tight_layout()
plt.show()
