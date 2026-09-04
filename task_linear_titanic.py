import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Convert gender into binary
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked into numerical values
data['Embarked'] = data['Embarked'].map({
    'S': 0,
    'C': 1,
    'Q': 2
})

# Feature selection
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']] #target
Y = data['Fare']   # Output / Target

# Handle missing values
X['Age'] = X['Age'].fillna(X['Age'].mean())
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

# Training and testing data split
# 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Display training and testing data size
print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = mse ** 0.5
r2 = r2_score(Y_test, Y_pred)

print("\nLinear Regression Results")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Graph
plt.scatter(Y_test, Y_pred)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Linear Regression - Titanic Dataset")
plt.show()