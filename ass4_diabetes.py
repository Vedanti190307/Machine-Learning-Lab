import pandas as pd
import numpy as mp

#lib for graph
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.model_selection import train_test_split #train test data split
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score

#load dataset
data=pd.read_csv("diabetes_risk_prediction_dataset.csv")

#feature selection 
X=data[['Age','Gender',]] #input
Y=data['Diabetes_risk']  #output

#training testing data split by hold out method 20%test and 80%train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#see how many rows and columns data have
print("\nTraining data: ",X_train.shape);
print("\nTesting data: ",X_test.shape)

#create logistic regression model
model=LogisticRegression(max_iter=1000)

#train model
model.fit(X_train,Y_train)

#prediction for testing data
Y_pred=model.predict(X_test)

#Evaluation Metrics
# 1-accuracy
accuracy=accuracy_score(Y_test,Y_pred)
print("\nAccuracy: ",accuracy)

# 2-precision
precision=precision_score(Y_test,Y_pred)
print("\nPrecision: ",precision)

# 3-confusion matrix
confusion_m=confusion_matrix(Y_test,Y_pred)
print("\nConfusion Matrix: ",confusion_m)

# 4-recall
recall=recall_score(Y_test,Y_pred)
print("\nRecall: ",recall)

# 5- f1-score
f1_score=f1_score(Y_test,Y_pred)
print("\nF1-Score: ",f1_score)

# 6-classification Report
classi_report=classification_report(Y_test,Y_pred)
print("Classificatin Report: ",classi_report)

#graph for confusion matrix
c_m=confusion_matrix(Y_test,Y_pred)

plt.figure(figsize=(5,4)) #create graph

#display confusion matrix as a heatmap
sns.heatmap(
            cm,       
            annot=True,   #displays value inside boxes
            fmt='d',      #displays value as integer
            cmap="blues"  #color style
            )



# Add title and labels
plt.title("Confusion Matrix") #title of graph
plt.xlabel("Predicted") #name of x axis
plt.ylabel("Actual")    #name for y axis

plt.show() # Display graph



