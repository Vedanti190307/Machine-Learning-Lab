#import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,recall_score,f1_score,classification_report

#load dataset
data=pd.read_csv("Breast_Cancer.csv")

print("First 5 Rows:") #display 1st 5 rows
print(data.head())

print("\nMissing Values:") # isnull()=check missing values in each column
print(data.isnull().sum()) #sum()=count missing values in each column

#feature selection
X=data[['Age','Tumor Size','Regional Node Examined','Reginol Node Positive','Survival Months']]
Y=data['Status']

#training testing data split by hold-out method 20%=test and 80%=train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# SVM Model
model = SVC(kernel='linear')

#train model
model.fit(X_train,Y_train)

#predicting output for testing data
Y_pred=model.predict(X_test)

#Evaluation metrics
# 1-accuracy
accuracy=accuracy_score(Y_test,Y_pred)
print("\nAccuracy: ",accuracy)

#2-precision
precision=precision_score(Y_test,Y_pred,pos_label='Alive')
print("\nPrecision :",precision)

#3-confusion matrix
c_m=confusion_matrix(Y_test,Y_pred)
print("\nConfusion Matrix: ",c_m)

#4-Recall Score
recall=recall_score(Y_test,Y_pred,pos_label='Alive')
print("\nRecall: ",recall)

#5- f1 score
f1=f1_score(Y_test,Y_pred,pos_label='Alive')
print("\nF1 Score: ",f1)

#6= classification report
classi_report=classification_report(Y_test,Y_pred)
print("\nClassification Report: ",classi_report)