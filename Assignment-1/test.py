from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np

preprocessor = MyPreProcessor()

print('Linear Regression')

X, y = preprocessor.pre_process(0)
linear = MyLinearRegression(iters=500,learning_rate=0.5)
linear.fit(X, y,errorType="rmse",folds=10)
# linear.normal_form(X,y)

# ypred = linear.predict(Xtest)

# print('Predicted Values:', ypred)
# print('True Values:', ytest)

print('Logistic Regression')

X, y = preprocessor.pre_process(2)
logistic = MyLogisticRegression(iters=1000,learning_rate=0.01)
logistic.fit(X, y,gradientType="bgd")

# ypred = logistic.predict(Xtest)

# print('Predicted Values:', ypred)
# print('True Values:', ytest)