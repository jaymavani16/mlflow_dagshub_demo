import mlflow.sklearn
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
import mlflow

# load data
iris = load_iris()
X = iris.data
y = iris.target

#performing train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fixing parameters
max_depth = 5
# apply mlflow
mlflow.set_experiment('iris-dt')
with mlflow.start_run(run_name="practice-artifacts-code-model"):
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train,y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_pred,y_test)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)

   

    #log confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion_matrixs')

    #save the confusion matrix
    plt.savefig('confusion_matrix.png')

    #log artifacts
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt,'Decisioin_tree')
    print('accuracy',accuracy)


