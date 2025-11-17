import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
file_path="C:/Users/ADMIN 21/Downloads/heart (1).csv"
data=pd.read_csv(file_path) 
X=data.iloc[:,:-1] 
Y=data.iloc[:,-1] 
data.info() 
X=data.drop(columns="output") 
Y=data['output'] 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42) 
model=DecisionTreeClassifier(criterion="entropy",random_state=42)  
model.fit(X_train,Y_train) 
Y_Pred=model.predict(X_test) 
accuracy=accuracy_score(Y_test, Y_Pred) 
print(f"Accuracy={accuracy*100:2f}%")