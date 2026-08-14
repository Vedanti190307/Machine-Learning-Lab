import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("car data.csv")

# Convert Onroad Price to number
df["Onroad Price"] = df["Onroad Price"].str.replace(",", "").astype(float)

# Convert Type to numeric
df["Type"] = df["Type"].str.strip()
df["Type"] = df["Type"].replace({"Petrol":1,"Diesel":0})

# Features
X = df[["CC","Seats","Variants"]]

# Target
y = df["Onroad Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("R2 Score:", model.score(X_test, y_test))

# Graph
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression")
plt.show()