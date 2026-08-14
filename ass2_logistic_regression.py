import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("car data.csv")

# ----------------------------
# Data Preprocessing
# ----------------------------

# Remove commas from price and convert to float
df["Onroad Price"] = (
    df["Onroad Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Convert Type into numeric
# Petrol = 1, Diesel = 0
df["Type"] = (
    df["Type"]
    .str.strip()
    .map({"Petrol": 1, "Diesel": 0})
)

# ----------------------------
# Features and Target
# ----------------------------
X = df[["CC", "Seats", "Variants", "Onroad Price"]]
y = df["Type"]

# ----------------------------
# Split Dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Train Model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# Evaluation Metrics
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_binary = f1_score(y_test, y_pred, average="binary")
f1_macro = f1_score(y_test, y_pred, average="macro")

print("Accuracy           :", round(accuracy, 4))
print("Precision          :", round(precision, 4))
print("Recall             :", round(recall, 4))
print("F1 Score (Binary)  :", round(f1_binary, 4))
print("F1 Score (Macro)   :", round(f1_macro, 4))

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Diesel", "Petrol"]
))

# ----------------------------
# Dashboard (2 Graphs in One Window)
# ----------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Graph 1 : Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Diesel", "Petrol"]
)

disp.plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Confusion Matrix")

# Graph 2 : Actual vs Predicted
ax[1].plot(
    range(len(y_test)),
    y_test.values,
    "bo-",
    linewidth=2,
    markersize=8,
    label="Actual"
)

ax[1].plot(
    range(len(y_pred)),
    y_pred,
    "r*-",
    linewidth=2,
    markersize=10,
    label="Predicted"
)

ax[1].set_xlabel("Sample Number")
ax[1].set_ylabel("Class (0 = Diesel, 1 = Petrol)")
ax[1].set_title("Actual vs Predicted")
ax[1].legend()
ax[1].grid(True)

plt.suptitle("Logistic Regression Dashboard", fontsize=16)
plt.tight_layout()

plt.show()