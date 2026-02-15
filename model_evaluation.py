import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load test data
df = pd.read_csv("test_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Scale features (same as training)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Load trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X)

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))