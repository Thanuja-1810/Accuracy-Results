import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix 
file_path="C:/Users/ADMIN 21/Downloads/heart (1).csv"
df=pd.read_csv(file_path) 
x=df.iloc[:,:-1].values 
y=df.iloc[:,-1].values 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size
 =0.2,random_state=42) 
model=GaussianNB() 
model.fit(x_train,y_train) 
y_pred=model.predict(x_test) 
print("Accuracy:",accuracy_score(y_test,y_pred)) 
print("Classification Report:\n",classification_report(y_test,y_pred)) 
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred)) 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
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
model=RandomForestClassifier(n_estimators=10,random_state=42)  
model.fit(X_train,Y_train) 
Y_Pred=model.predict(X_test) 
accuracy=accuracy_score(Y_test, Y_Pred) 
print(f"Accuracy={accuracy*100:2f}%")