import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Q2():
    def __init__(self):
        self.dataset=pd.read_csv('weight-height.csv')
        self.Xtrain=None
        self.Ytrain=None
        self.Xtest=None
        self.Ytest=None

    def preProcess(self):
        Y=self.dataset.iloc[:,2].to_numpy()
        X=self.dataset.iloc[:,1].to_numpy()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=4, train_size = 0.8)


    def bootstrap(self,X,Y):
        # data=np.concatenate((X,Y),axis=1)
        data=np.column_stack((X,Y))
        data=pd.DataFrame(data)
        indexes=np.random.choice(len(X),replace=True,size=len(X))
        new_data=data.iloc[indexes]
        X=new_data.iloc[:,0].to_numpy()
        Y=new_data.iloc[:,1].to_numpy()
        return X,Y

    def predict(self,samples):
        hi=[]
        for i in range(samples):
            X_boot,Y_boot=self.bootstrap(self.X_train,self.Y_train)
            X_boot=X_boot.reshape(-1, 1)
            reg = LinearRegression().fit(X_boot, Y_boot)
            self.X_test=self.X_test.reshape(-1,1)
            Y_pred=reg.predict(self.X_test)
            hi.append(np.mean(Y_pred,axis=0)) #mean of predicted y
        hi=np.array(hi)
        hBar=np.mean(hi,axis=0)  #mean of mean of predicted Ys
        # print(hBar)
        yBar=np.mean(self.Y_test,axis=0)   #actual y mean
        # print(yBar)
        bias=(hBar-yBar)
        print("Number of samples = ",samples)
        print("Bias = ",bias)
        var=0
        mse=0
        for i in hi:
            var+=(hBar-i)**2
            mse+=(yBar-i)**2
        n=len(self.X_train)
        var/=(samples-1)
        mse/=samples
        print("Variance = ",var)
        print("MSE = ",mse)
        print("Value = ",(mse-(bias**2)-var))


        


        
Q=Q2()
Q.preProcess()
Q.predict(100)


