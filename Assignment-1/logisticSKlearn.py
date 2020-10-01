from sklearn.linear_model import LogisticRegression
from scratch import MyLogisticRegression, MyPreProcessor
import numpy as np

preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(2)
logistic = MyLogisticRegression(iters=1000,learning_rate=0.01)
Xtrain,Ytrain,Xtest,Ytest,Xval,Yval=logistic.divide(X,y)

skModel=LogisticRegression(max_iter=1000,random_state=432)
skModel.fit(Xtrain,Ytrain)
acc=skModel.score(Xtrain,Ytrain)
print(acc)