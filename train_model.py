import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

# ===== Load Dataset =====
df = pd.read_csv("heart.csv")

# ===== One-Hot Encoding for categorical features =====
categorical_cols = ["cp", "restecg", "slope", "thal"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ===== Split Features and Target =====
X = df.drop("target", axis=1)
y = df["target"]

# ===== Balance Dataset using SMOTE =====
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# ===== Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# ===== Feature Scaling =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===== Logistic Regression =====
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

# ===== Predictions =====
y_pred = model.predict(X_test)

# ===== Metrics =====
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)  # Recall for class 1 (Heart Disease)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"Sensitivity (Recall): {sensitivity*100:.2f}%")
print(f"Specificity: {specificity*100:.2f}%")

# ===== Save Model & Scaler =====
pickle.dump(model, open("heart_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(list(X.columns), open("model_features.pkl", "wb"))
print("Final Model and Scaler saved successfully!")
