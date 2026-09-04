#import libraries
import pandas as pd
import numpy as np

#lib for graph
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.model_selection import train_test_split #test train split
from sklearn.linear_model import LogisticRegression  #logistic regression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, classification_report


#load dataset 
data=pd.read_csv("Titanic-Dataset.csv")
# 3. READ / DISPLAY DATASET
print("First 5 Rows:") #display 1st 5 rows
print(data.head())

print("\nDataset Shape:") #column rows
print(data.shape)

print("\nColumns:") #display all column names
print(data.columns)

print("\nMissing Values:") # isnull()=check missing values in each column
print(data.isnull().sum()) #sum()=count missing values in each column

#convet sex into binary male=0 and feamle=1
data['Sex']=data['Sex'].map({'male':0,'female':1})

#feature selection
X=data[['Pclass','Sex','Age','SibSp','Parch','Fare']] #input
Y=data['Survived']                                    #output

#Handle missing values
X['Age']=X['Age'].fillna(X['Age'].mean())

#training testing data split by hold-out method 20%=test and 80%=train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# see how many rows and columns data have
print("\nTraining Data: ",X_train.shape);
print("\nTesting Data: ",X_test.shape)

#create logistic regression model
model=LogisticRegression(max_iter=1000)

#train model
model.fit(X_train,Y_train)

#prediction for testing data
Y_pred=model.predict(X_test)

#Evaluation Metrics
# 1-accuracy=overall correct predictions
accuracy=accuracy_score(Y_test,Y_pred)
print("\nAccuracy: ",accuracy)

#2-precision=correct positive prediction
precision=precision_score(Y_test,Y_pred)
print("\nPrecision: ",precision)

#3-confusion matrix=
confusion_m=confusion_matrix(Y_test,Y_pred)
print("\nConfusion Matrix: ",confusion_m)

# 4-recall
recall = recall_score(Y_test,Y_pred,average='weighted',zero_division=0)
print("\nRecall: ", recall)

# 5- f1-score
f1 = f1_score(Y_test,Y_pred,average='weighted',zero_division=0)
print("\nF1-Score: ", f1)

#6-classification report=
classi_report=classification_report(Y_test,Y_pred)
print("Classification Report",classi_report)

#graph for confysion matrix
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(5, 4))  # Create graph

# Display confusion matrix as a heatmap
sns.heatmap(
    cm,
    annot=True,      # Display values inside boxes
    fmt='d',         # Display values as integers
    cmap='Blues'     # Color style
)

# Add title and labels
plt.title("Confusion Matrix") #title of graph
plt.xlabel("Predicted") #name of x axis
plt.ylabel("Actual")    #name for y axis

plt.show() # Display graph 