import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os
import numpy as np

# --- Model Training ---

# Load the dataset
try:
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: 'heart.csv' not found at path {data_path}.")
    exit()

# --- Full Preprocessing Pipeline ---

# 1. Replace placeholder '?' with actual NaN
data.replace('?', np.nan, inplace=True)
print("Replaced '?' with NaN for missing value handling.")

# 2. Handle the target column ('num') first
if 'num' not in data.columns:
    print("Error: The required diagnosis column 'num' was not found.")
    exit()
# Convert target to binary (0 for no disease, 1 for disease)
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)
target_column = 'num'
print(f"Using '{target_column}' as the target column.")

# 3. Separate features (X) from target (y)
y = data[target_column]
X = data.drop(columns=[target_column, 'id', 'dataset'], errors='ignore') # Drop target and unnecessary cols

# 4. Convert all categorical columns to numeric using One-Hot Encoding
print("Applying One-Hot Encoding to all non-numeric columns...")
X = pd.get_dummies(X, drop_first=True) # drop_first helps prevent multicollinearity

# 5. Handle any remaining missing values AFTER encoding
print("Checking for and filling any remaining missing values...")
for col in X.columns:
    if X[col].isnull().any():
        median = X[col].median()
        # This syntax avoids the FutureWarning
        X[col] = X[col].fillna(median)
        print(f"Filled missing values in '{col}' with median: {median}")

# --- End Preprocessing ---

# Split the fully preprocessed data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
print("\nTraining the model...")
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model at the demo screening threshold. This is a development
# metric only; it does not establish clinical performance or safety.
screening_threshold = 0.30
y_scores = model.predict_proba(X_test)[:, 1]
y_pred = (y_scores >= screening_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
sensitivity = tp / (tp + fn) if tp + fn else 0
specificity = tn / (tn + fp) if tn + fp else 0
print(f"Accuracy at {screening_threshold:.0%} demo threshold: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Sensitivity: {sensitivity * 100:.2f}% | Specificity: {specificity * 100:.2f}%")

# Save both the model and the columns it was trained on
model_data = {
    "model": model,
    "columns": X_train.columns.tolist(),
    "threshold": screening_threshold,
}
model_output_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
with open(model_output_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Model, training columns, and demo threshold saved to {model_output_path}.")
