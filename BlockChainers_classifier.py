from os import pread
import sklearn
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy import stats as st
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import math
import sys
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

"""##***Utility functions***"""

inp=sys.argv[1]

def preprocess(data):
  le1 = preprocessing.LabelEncoder()
  le1.fit(data["Island"])

  le2=preprocessing.LabelEncoder()
  le2.fit(data["Clutch Completion"])

  le3=preprocessing.LabelEncoder()
  le3.fit(data["Sex"])

  le4=preprocessing.LabelEncoder()
  le4.fit(data["Species"])

  return (le1,le2,le3,le4)


def test_nan(df):
  null_vals=[]
  nulls=df.isnull()
  for i in range(len(nulls)):
    for j in range(len(nulls.iloc[i,:])):
      if(nulls.iloc[i,j]):
        null_vals.append((i,j))
  return null_vals


def replace_nan(data,null_vals,discrete):
  for i in null_vals:
    if(i[1] in discrete):
      mode=data[data.columns[i[1]]].mode()[0]
      data.iloc[i[0],i[1]]=mode
    else:
      mean=data[data.columns[i[1]]].mean()
      data.iloc[i[0],i[1]]=mean
  return data


def accuracy(pred,actual):
  accu=0
  for i in range(len(pred)):
    accu+=(pred[i]==actual[i])
  # print(accu/len(ytest))
  return accu/len(actual)

def normalize_attributes(columns,df):
  scalers=[]
  scaler = MinMaxScaler()
  scaler.fit(df.iloc[:,columns])
  vals=scaler.transform(df.iloc[:,columns])
  df.iloc[:,columns]=vals
  return (scaler,vals)

def voting(predictions):
  predictions=np.array(predictions)
  predictions=predictions.transpose()
  preds=[]
  for i in predictions:
    preds.append(st.mode(i)[0][0])
  return preds

def preprocess_test_data(df,le1,le2,le3,le4):
  df["Island"]=le1.transform(df["Island"])
  df["Clutch Completion"]=le2.transform(df["Clutch Completion"])
  df["Sex"]=le3.transform(df["Sex"])
  return df

def feature_engineering(testdf,le1,le2,le3,le4,scaler1,scaler2,pca,scaled):  
  testdf=preprocess_test_data(testdf,le1,le2,le3,le4)
  testdf.iloc[:,scaled]=scaler1.transform(testdf.iloc[:,scaled])
  xtest=testdf.iloc[:,:].values
  pcom=pca.transform(testdf.iloc[:,:])
  principalDf = pd.DataFrame(data = pcom,  columns = ['principal component {}'.format(i+1) for i in range(n_components)])
  testdf=principalDf
  testdf.iloc[:,:]=scaler2.transform(testdf.iloc[:,:])
  xtest=testdf.iloc[:,:].values
  return testdf

"""## **Pre processing**"""

df=pd.read_csv("penguins_train.csv")
cols=list(df.columns)
discrete=[cols.index("Island"),cols.index("Clutch Completion"),cols.index("Sex"),cols.index("Species")]

null_vals=test_nan(df)
# print(null_vals)
df=replace_nan(df,null_vals,discrete)
null_vals=test_nan(df)
# print(null_vals)
for i in range(len(df)):
  if(df.iloc[i,cols.index("Sex")]=="."):
    df.drop([i],inplace=True)
    break
le1,le2,le3,le4=preprocess(df)
df = df.reset_index(drop=True)
# print(df)
X_train, y_train = df.iloc[:,:-1].values,df.iloc[:,-1:]
train=pd.DataFrame(X_train,columns=cols[:-1])
train["Island"]=le1.transform(train["Island"])
train["Clutch Completion"]=le2.transform(train["Clutch Completion"])
train["Sex"]=le3.transform(train["Sex"])
train[cols[-1]]=y_train
train["Species"]=le4.transform(train["Species"])
xtrain=train.iloc[:,:-1].values
ytrain=train.iloc[:,-1].values

oversample = SMOTE()
x, y = oversample.fit_resample(xtrain, ytrain)
train=pd.DataFrame(x,columns=cols[:-1])
train[cols[-1]]=y

scaled=[cols.index("Culmen Length (mm)"),cols.index("Culmen Depth (mm)"),cols.index("Flipper Length (mm)"),cols.index("Body Mass (g)"),cols.index("Delta 15 N (o/oo)"),cols.index("Delta 13 C (o/oo)")]
scaler1,vals=normalize_attributes(scaled,train)
train.iloc[:,scaled]=vals
xtrain=train.iloc[:,:-1].values
ytrain=train.iloc[:,-1].values

n_components=5
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(xtrain)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component {}'.format(i+1) for i in range(n_components)])
principalDf[cols[-1]]=train.iloc[:,-1:]
train=principalDf

scaler2,vals=normalize_attributes(range(len(train.columns))[:-1],train)
train.iloc[:,:-1]=vals
xtrain=train.iloc[:,:-1].values
ytrain=train.iloc[:,-1].values



"""## ***All vs All classifier ***"""
classes=le4.classes_
labels=le4.transform(classes)
allpairs=list(combinations(labels,2))

def all_vs_all(train,classifier,xtest):
  allpred=[]
  for i in range(len(allpairs)):
    newdf=train[(train.Species==allpairs[i][0]) | (train.Species==allpairs[i][1])]
    x=newdf.iloc[:,:-1].values
    y=newdf.iloc[:,-1:].values

    model=svm.SVC(kernel="rbf").fit(x, y)

    if(classifier=="SVM"):
      model=svm.SVC(kernel="rbf").fit(x, y)
    if(classifier=="LogisticRegression"):
      model = LogisticRegression(random_state=0).fit(x, y)
    if(classifier=="RandomForest"):
      model=RandomForestClassifier(max_depth=4, random_state=0)
    if(classifier=="DecisionTree"):
      model=tree.DecisionTreeClassifier()
    if(classifier=="KNN"):
      model=KNeighborsClassifier(n_neighbors=3)

    model.fit(x,y)
    pred=model.predict(xtest)
    allpred.append(pred)
  return allpred


best_model="SVM"
testdf=pd.read_csv(inp)

null_vals=test_nan(testdf)
testdf=replace_nan(testdf,null_vals,discrete)
null_vals=test_nan(testdf)

testdf=feature_engineering(testdf,le1,le2,le3,le4,scaler1,scaler2,pca,scaled)
allpred=all_vs_all(train,"SVM",testdf)
pred=voting(allpred)
# print(pred)
trans_pred=le4.inverse_transform(pred)
# print(trans_pred)
predicted=pd.DataFrame(trans_pred,columns=["Predicted_labels"])
# predicted.set_index("Predicted_labels",inplace=True)
predicted.to_csv("predicted_labels.csv")