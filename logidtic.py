import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix 
file_path="C:/Users/ADMIN 21/Downloads/heart (1).csv"
df=pd.read_csv(file_path) 
x=df.iloc[:,:-1].values 
y=df.iloc[:,-1].values 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) 
model=LogisticRegression(max_iter=1000,random_state=42) 
model.fit(x_train,y_train) 
y_pred=model.predict(x_test) 
print("Accuracy:",accuracy_score(y_test,y_pred)) 
print("Classification Report:\n",classification_report(y_test,y_pred)) 
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred)) 