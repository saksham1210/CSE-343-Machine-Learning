import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns




class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self,learning_rate=0.05,iters=500):
        self.learning_rate=learning_rate
        self.iters=iters
        self.theta=None


    def sigmoid(self,z):
        z=np.array(z,dtype=np.float32)
        ans=1/(1+np.exp(-z))
        return ans

    '''COST FUNCTION'''
    def cost(self,X,y,theta):
        n=len(X)
        y_pred=self.sigmoid(X.dot(theta))
        cost= -np.sum((y*np.log(y_pred))+((1-y)*np.log(1-y_pred)))/n
        return cost

    ''' BATCH GRADIENT DESCENT'''
    def BGD(self,X,y,theta,learningRate,iters):
        cost_history=[]
        n=len(y)
        for _ in range(iters):
            predic=self.sigmoid(X.dot(theta))
            theta=np.add(theta,learningRate*(X.T.dot((y-predic))/n))
            cost=self.cost(X,y,theta)
            cost_history.append(cost)
        
        return cost_history,theta

    '''PLOTTING THE GRAPHS'''
    def Plot(self,arr1,x_label,y_label):
        print("PLotting the curves")
        plt.plot(arr1)
        plt.xlabel(x_label) 
        plt.ylabel(y_label) 
        plt.show()


    # def EDA():



    def fit(self, X, y):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        n_samples,n_features =X.shape
        X=np.insert(X,0,np.ones(n_samples),axis=1) #adding a column of 1 at starting of X matrix for theta0 the bias
        n_samples,n_features =X.shape
        theta=np.zeros(n_features)
        train_cost_history=[]
        train_cost_history,self.theta=self.BGD(X,y,theta,self.learning_rate,self.iters)
        print("Training Loss=",train_cost_history[-1])   
   
        self.Plot(train_cost_history,"Iterations","Error")
        self.accuracy(self.predict(X),y)
        print(self.theta)
        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        y_pred=self.sigmoid(X.dot(self.theta))

        # return the numpy array y which contains the predicted values
        return np.round(y_pred)

    def accuracy(self,y_pred,y):
        correct=0
        for i in range(len(y)):
            if y[i]==y_pred[i]:
                correct+=1
        print("Accuracy=",(correct/len(y))*100)
        return correct/len(y)





df_2=pd.read_csv("Q4_Dataset.txt",sep="   ",header=None)
# print(df_2)
df_2=df_2.sample(frac=1,random_state=43).reset_index(drop=True)
df_2=df_2.apply(pd.to_numeric, errors='coerce')
df_2=df_2.dropna()
X=df_2.iloc[:,1:]
y=df_2.iloc[:,0]
X=X.to_numpy()
y=y.to_numpy()
logistic = MyLogisticRegression(iters=100000,learning_rate=0.0002)
logistic.fit(X, y)
