# Run this to identify potential leakage
import pandas as pd
import numpy as np

df = pd.read_csv('../data/processed/preprocessed_features.csv')

# Check for perfect correlation with target
corr_with_target = df.corr()['mortality'].abs().sort_values(ascending=False)
print("Top correlations with mortality:")
print(corr_with_target.head(10))

# Inspect features that might encode outcome information
suspicious_features = [col for col in df.columns if any(term in col.lower() for term in 
                      ['death', 'expire', 'mortality', 'outcome', 'survival', 'died'])]
print("Suspicious feature names:", suspicious_features)

# 1. Check for duplicate or near-duplicate rows
print("Checking for duplicates...")
dup_count = df.duplicated().sum()
print(f"Found {dup_count} exact duplicates")

# 2. Check your train-test split implementation
from sklearn.model_selection import train_test_split

# Use a different random state to verify
X = df.drop('mortality', axis=1)
y = df['mortality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Try a simple model with this new split
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Score with new split: {model.score(X_test, y_test)}")

# 3. Try a very simple model to see if it also gets perfect results
from sklearn.linear_model import LogisticRegression
simple_model = LogisticRegression(max_iter=1000)
simple_model.fit(X_train, y_train)
print(f"Logistic regression score: {simple_model.score(X_test, y_test)}")