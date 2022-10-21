import pandas as pd
def accuracy(pred,actual):
  accu=0
  for i in range(len(pred)):
    accu+=(pred[i]==actual[i])
  # print(accu/len(ytest))
  return accu/len(actual)

dfact=pd.read_csv("sample_result.csv")
dfpred=pd.read_csv("predicted_labels.csv")
act=dfact["Species"]
pred=dfpred["Predicted_labels"]
print(accuracy(pred,act))