import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        self.df_0=pd.read_csv('Dataset_abalone.data',sep=" ",header=None)
        self.df_1=pd.read_csv('VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv')
        self.df_2=pd.read_csv("data_banknote_authentication.txt",sep=",",header=None)


    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        # np.empty creates an empty array only. You have to replace this with your code.
        X = np.empty((0,0))
        y = np.empty((0))

        if dataset == 0:
            # Implement for the abalone dataset
            # M==0   F==1   I==2
            self.df_0=self.df_0.sample(frac=1,random_state=4).reset_index(drop=True)
            self.df_0.iloc[:,0].replace("M","0",inplace=True)
            self.df_0.iloc[:,0].replace('F',"1",inplace=True)
            self.df_0.iloc[:,0].replace('I',"2",inplace=True)
            self.df_0=self.df_0.apply(pd.to_numeric, errors='coerce')
            self.df_0=self.df_0.dropna()
            X=self.df_0.iloc[:,:-1]
            y=self.df_0.iloc[:,-1]
            X=X.to_numpy()
            y=y.to_numpy()
        elif dataset == 1:
            # Implement for the video game dataset
            self.df_1=self.df_1.sample(frac=1,random_state=4).reset_index(drop=True)
            self.df_1=self.df_1.filter(['Critic_Score','User_Score','Global_Sales'])
            self.df_1=self.df_1.apply(pd.to_numeric, errors='coerce')
            self.df_1=self.df_1.dropna()
            X=self.df_1.filter(['Critic_Score','User_Score'])
            y=self.df_1.filter(['Global_Sales'])
            X=X.to_numpy()
            y=y.to_numpy()
        elif dataset == 2:
            # Implement for the banknote authentication dataset
            self.df_2=self.df_2.sample(frac=1,random_state=43).reset_index(drop=True)
            self.df_2=self.df_2.apply(pd.to_numeric, errors='coerce')
            self.df_2=self.df_2.dropna()
            X=self.df_2.iloc[:,:-1]
            y=self.df_2.iloc[:,-1]
            X=X.to_numpy()
            y=y.to_numpy()
            # self.EDA(self.df_2)
        return X, y


    def EDA(self, df):
        # print(df)
        print("Pairwise correlation of features, using standard correlation coefficient")
        print(df.corr(method='pearson'))  # correlation matrix
        
        print(" ")
        print("Info")
        print(df.info()) # attribute type info
        print("")
        print("Data Description")
        print(df.describe())  #Gives description of each column
        print("")
        sns.countplot(x=4,data=df) #For histogram of class distribution
        plt.title('Different Classes')
        plt.show()
        print(df.iloc[:,-1].value_counts())
        print("")
        df.hist(figsize=(11,6), grid =False, layout = (2,4), bins=100) #histogram of attribute distributions
        plt.show()
        print("")
        sns.pairplot(df,hue=4) #Used to make pairplots 
        plt.title("PairPlot")
        plt.show()







class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """

    def __init__(self,learning_rate=0.05,iters=500):
        self.learning_rate=learning_rate
        self.iters=iters
        self.theta=None
        self.train_cost=float('inf')
        self.train_cost_history=None
        self.test_cost_history=None
        self.test_cost=float('inf')
        self.avg_K_error=0
        self.bestfold=0
        
    def cost_rmse(self,X,y,theta):
        n=len(y)
        err=0
        for i in range(n):
            err+=(((X[i].dot(theta))-y[i])**2)
        return math.sqrt(err/(n))


    def cost_mae(self,X,y,theta):
        n=len(y)
        err=0
        for i in range(n):
            err+=abs((X[i].dot(theta))-y[i])
        return err/n
        

    def gradient_descent_rmse(self,X,y,theta,learningRate):
        n_samples,n_features=X.shape
        dtheta=np.zeros(n_features)
        denominator=0
        for i in range(n_samples):
            denominator += ((X[i].dot(theta))-y[i])**2
        denominator=(denominator*n_samples)**(0.5)
        for j in range(n_samples):    # 0 to m
            dtheta=np.add(dtheta , ((X[j].dot(theta)-y[j])*X[j])/denominator)
        theta=theta-(learningRate*dtheta)
        return theta 


    def gradient_descent_mae(self,X,y,theta,learningRate):
        n_samples,n_features=X.shape
        dtheta=np.zeros(n_features)
        for j in range(n_samples):    # 0 to m
            dtheta=np.add(dtheta , (np.sign(X[j].dot(theta)-y[j])*X[j])/n_samples)
        theta=theta-(learningRate*dtheta)
        return theta 

    '''FOR DIVIDING THE DATASET INTO K FOLDS'''

    def k_divide(self,k,folds_x,folds_y):
        Xtrain=folds_x.copy()
        Xtest=np.array(folds_x[k],dtype=object)
        del Xtrain[k]
        Xtrain=np.array(Xtrain,dtype=object)
        l=len(Xtrain)
        Xtrain_final=Xtrain[0]
        for j in range(1,l):
            Xtrain_final=np.concatenate((Xtrain_final,Xtrain[j]))
        Ytrain=folds_y.copy()
        Ytest=np.array(folds_y[k],dtype=object)
        del Ytrain[k]
        Ytrain=np.array(Ytrain,dtype=object)
        l=len(Ytrain)
        Ytrain_final=Ytrain[0]
        for j in range(1,l):
            Ytrain_final=np.concatenate((Ytrain_final,Ytrain[j])) 
        return Xtrain_final,Ytrain_final,Xtest,Ytest

    '''NORMAL FORM OF LINEAR REGRESSION'''
    def normal_form(self,X,y):
        folds_x=np.array_split(X,10)
        folds_y=np.array_split(y,10)
        Xtrain_final,Ytrain_final,Xtest,Ytest=self.k_divide(9,folds_x,folds_y)
        self.theta=np.dot(np.linalg.inv((np.dot(Xtrain_final.T,Xtrain_final))),np.dot(Xtrain_final.T,Ytrain_final))
        test_cost=self.cost_mae(Xtest,Ytest,self.theta)
        train_cost=self.cost_mae(Xtrain_final,Ytrain_final,self.theta)
        print("Test Cost=",test_cost)
        print("Train Cost=",train_cost)
        return self

    def fit(self, X, y,errorType="rmse",folds=5):
        """
        Fitting (training) the linear model.

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
        # for folds in range(2,11):

        ''' K-Fold Cross validation '''
        avg_K_error=0
        print(str(folds)+"-Fold Cross Validation")
        folds_x=np.array_split(X,folds)
        folds_y=np.array_split(y,folds)
        for k in range(folds):
            Xtrain_final,Ytrain_final,Xtest,Ytest=self.k_divide(k,folds_x,folds_y)

            '''Training the data'''

            train_cost_history=[]
            test_cost_history=[]
            n_samples,n_features=X.shape
            theta=np.zeros(n_features)
            train_cost=0
            test_cost=0
            if (errorType=="rmse"):
                for _ in range(self.iters):
                    theta=self.gradient_descent_rmse(Xtrain_final,Ytrain_final,theta,self.learning_rate)
                    train_cost=self.cost_rmse(Xtrain_final,Ytrain_final,theta)
                    train_cost_history.append(train_cost)
                    test_cost=self.cost_rmse(Xtest,Ytest,theta)
                    test_cost_history.append(test_cost)
            else:
                for _ in range(self.iters):
                    theta=self.gradient_descent_mae(Xtrain_final,Ytrain_final,theta,self.learning_rate)
                    train_cost=self.cost_mae(Xtrain_final,Ytrain_final,theta)
                    train_cost_history.append(train_cost)
                    test_cost=self.cost_mae(Xtest,Ytest,theta)
                    test_cost_history.append(test_cost)
            train_cost_history=np.array(train_cost_history)
            # print("theta==",theta)
            print(("Div-"+str(k)+" Testing Cost = "),test_cost)
            avg_K_error+=test_cost
            if test_cost<self.test_cost:
                self.train_cost=train_cost
                self.train_cost_history=train_cost_history
                self.test_cost_history=test_cost_history
                self.test_cost=test_cost
                self.bestfold=k
        self.avg_K_error=avg_K_error/folds
        print(("AVG "+errorType+" Error for total "+str(folds)+" folds = "),self.avg_K_error)
        print("Best fold=",self.bestfold," with test error=",self.test_cost)
            # print("------------------------------------------------------------------------------------")
        self.Plot(train_cost_history,test_cost_history,"Train loss","Test loss","Iterations","Error",errorType)
        
        # fit function has to return an instance of itself or else it won't work with test.py
        return self


    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        y_pred=np.dot(X,self.theta)
        
        # return the numpy array y which contains the predicted values
        return y_pred



    '''FUNCTION TO PLOT THE GRAPHS'''
    def Plot(self,arr1,arr2,label1,label2,x_label,y_label,title):
        print("PLotting the curves")
        plt.plot(arr1,label=label1)
        plt.plot(arr2,label=label2)
        plt.title(title)
        plt.xlabel(x_label) 
        plt.ylabel(y_label) 
        plt.legend()
        plt.show()



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
        z=z+0.000001
        ans=1/(1+np.exp(-z))
        return ans

    '''COST FUNCTION'''
    def cost(self,X,y,theta):
        n=len(X)
        y_pred=self.sigmoid(X.dot(theta))
        cost= -np.sum((y*np.log(y_pred+0.000001))+((1-y)*np.log(1-y_pred+0.000001)))/n
        return cost

    ''' BATCH GRADIENT DESCENT'''
    def BGD(self,X,y,Xval,Yval,Xtest,Ytest,theta,learningRate,iters):
        cost_history=[]
        cost_history_val=[]
        cost_history_test=[]
        n=len(y)
        for _ in range(iters):
            predic=self.sigmoid(X.dot(theta))
            theta=np.add(theta,learningRate*(X.T.dot((y-predic))/n))
            cost=self.cost(X,y,theta)
            cost_history.append(cost)
            cost_history_val.append(self.cost(Xval,Yval,theta))
            cost_history_test.append(self.cost(Xtest,Ytest,theta))
        return cost_history_test,cost_history,cost_history_val,theta

    '''STOCHASTIC GRADIENT DESCENT'''
    def SGD(self,X,y,Xval,Yval,Xtest,Ytest,theta,learningRate,iters):
        cost_history=[]
        cost_history_val=[]
        cost_history_test=[]
        for _ in range(iters):
            index=np.random.randint(0,len(X),1)[0] #picking a random sample from the dataset
            X_s=X[index,:]
            y_s=y[index]
            predic=self.sigmoid(X_s.dot(theta))
            theta=np.add(theta,learningRate*(X_s.T.dot(y_s-predic))) #updating theta
            cost=self.cost(X,y,theta)
            cost_history.append(cost)
            cost_history_val.append(self.cost(Xval,Yval,theta))
            cost_history_test.append(self.cost(Xtest,Ytest,theta))
        return cost_history_test,cost_history,cost_history_val,theta

    '''SPLITTING THE DATASET IN THE TRAIN:VAL:TEST 7:1:2'''
    def divide(self,X,y):  #7:1:2
        folds_x=np.array_split(X,10)
        folds_y=np.array_split(y,10)
        folds_x1=folds_x.copy()
        Xval=np.array(folds_x1[7],dtype=object)
        Xtest=np.concatenate((folds_x1[-1],folds_x1[-2]))
        del folds_x1[-1]
        del folds_x1[-1]
        del folds_x1[-1]
        folds_x1=np.array(folds_x1,dtype=object)
        l=len(folds_x1)
        Xtrain=folds_x1[0]
        for j in range(1,l):
            Xtrain=np.concatenate((Xtrain,folds_x1[j]))
        folds_y1=folds_y.copy()
        Yval=np.array(folds_y1[7],dtype=object)
        Ytest=np.concatenate((folds_y1[-1],folds_y1[-2]))
        del folds_y1[-1]
        del folds_y1[-1]
        del folds_y1[-1]
        folds_y1=np.array(folds_y1,dtype=object)
        l=len(folds_x1)
        Ytrain=folds_y1[0]
        for j in range(1,l):
            Ytrain=np.concatenate((Ytrain,folds_y1[j]))
        return Xtrain,Ytrain,Xtest,Ytest,Xval,Yval

    '''PLOTTING THE GRAPHS'''
    def Plot(self,arr1,arr2,label1,label2,x_label,y_label,title):
        print("PLotting the curves")
        plt.plot(arr1,label=label1)
        plt.plot(arr2,label=label2)
        plt.title(title)
        plt.xlabel(x_label) 
        plt.ylabel(y_label) 
        plt.legend()
        plt.show()


    # def EDA():



    def fit(self, X, y,gradientType="bgd"):
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
        Xtrain,Ytrain,Xtest,Ytest,Xval,Yval=self.divide(X,y)
        n_samples,n_features =X.shape
        theta=np.zeros(n_features)
        train_cost_history=[]
        val_cost_history=[]
        test_cost_history=[]
        if gradientType=="sgd":
            test_cost_history,train_cost_history,val_cost_history,self.theta=self.SGD(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,theta,self.learning_rate,self.iters)
        else:
            test_cost_history,train_cost_history,val_cost_history,self.theta=self.BGD(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,theta,self.learning_rate,self.iters)
        print("Training Loss=",train_cost_history[-1])   
        print("Test Loss=",test_cost_history[-1])     
        self.Plot(train_cost_history,val_cost_history,"Train loss","Validation loss","Iterations","Error","Gradient Type-"+gradientType)
        self.accuracy(self.predict(Xtest),Ytest)
        self.accuracy(self.predict(Xtrain),Ytrain)
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

