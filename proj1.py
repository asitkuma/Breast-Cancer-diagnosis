from statistics import mode
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle as pk
def buildModel(clean_data):
    X=clean_data.drop(["diagnosis"],axis=1)
    Y=clean_data["diagnosis"]
    print(X)
    #create object
    scaler=StandardScaler()
    X=scaler.fit_transform(X)#normalize the data. y we done need to do because already it is.
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    LR=LogisticRegression()
    LR.fit(X_train,Y_train)
    Y_predict=LR.predict(X_test)
    print("Accuracy:-",accuracy_score(Y_predict,Y_test))
    print("Classification report:_ ",classification_report(Y_test,Y_predict))
    return LR,scaler

def cleanData():
    df1=pd.read_csv(r"C:\Users\Asit\OneDrive\Desktop\b_cancer\data.csv")
    df1.drop(["id","Unnamed: 32"],axis=1,inplace=True)
    df1.replace("M",1,inplace=True)
    df1.replace("B",0,inplace=True)
    return df1


data=cleanData()
model,scaler=buildModel(data)
print(model)
with open('model.pkl','wb') as f:#model binary file(for import and export) wb for write binary.
    pk.dump(model,f)
with open('scaler.pkl','wb') as f:#scaler binary file.
    pk.dump(scaler,f)