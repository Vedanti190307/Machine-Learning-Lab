# import libraries
import pandas as pd
import numpy as np

# lib for graph
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Naive Bayes Classifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, classification_report

# load dataset
data = pd.read_csv("diabetes_risk_prediction_dataset.csv")

# feature selection
X = data[['Age','Height_cm','Weight_kg','BMI','Waist_Circumference_cm','Blood_Glucose','HbA1c','Fasting_Blood_Sugar','Insulin_Level','Blood_Pressure_Systolic','Blood_Pressure_Diastolic','Total_Cholesterol','HDL','LDL','Triglycerides','Heart_Rate','Exercise_Hours_Per_Week','Daily_Walking_Minutes','Sleep_Hours','Daily_Water_Intake_L','Diabetes_Risk_Score']]  #input
Y = data['Diabetes_Risk']    # output

# handle missing values
X = X.fillna(X.median())

# training testing data split by hold out method
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# see how many rows and columns data have
print("\nTraining data: ", X_train.shape)
print("\nTesting data: ", X_test.shape)

# create Naive Bayes model
model = GaussianNB()

# train model
model.fit(X_train, Y_train)

# prediction for testing data
Y_pred = model.predict(X_test)


# Evaluation Metrics
# 1-accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("\nAccuracy: ",accuracy)

# 2-precision
precision = precision_score( Y_test,Y_pred,average='weighted',zero_division=0)
print("\nPrecision: ",precision)

# 3-confusion matrix
confusion_m = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix: ",confusion_m)

# 4-recall
recall = recall_score(Y_test,Y_pred,average='weighted',zero_division=0)
print("\nRecall: ", recall)

# 5- f1-score
f1 = f1_score(Y_test,Y_pred,average='weighted',zero_division=0)
print("\nF1-Score: ", f1)

# 6-classification Report
classi_report = classification_report(Y_test,Y_pred,zero_division=0)
print("\nClassification Report: ")
print(classi_report)

# graph for confusion matrix
plt.figure(figsize=(5, 4))

# display confusion matrix as a heatmap
sns.heatmap(confusion_m,annot=True,fmt='d',cmap="Blues")

# Add title and labels
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()