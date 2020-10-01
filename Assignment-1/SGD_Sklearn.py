from sklearn.linear_model import SGDClassifier
from scratch import MyLogisticRegression, MyPreProcessor
import numpy as np

preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(2)
logistic = MyLogisticRegression(iters=1000,learning_rate=0.01)
Xtrain,Ytrain,Xtest,Ytest,Xval,Yval=logistic.divide(X,y)

skModel=SGDClassifier(loss="log", alpha=0.01,max_iter=1000)
skModel.fit(Xtrain,Ytrain)
acc=skModel.score(Xtest,Ytest)
print(acc)