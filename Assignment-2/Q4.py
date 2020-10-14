import h5py as h5
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

class NaiveBayes():
    
    def __init__(self):
        self.classes=None
        self.freq_classes=None
        self.subData={}
        self.mean_classes={}
        self.standard_dev={}

# Pre-processing Data
    def preProcess(self,Dataset):
        X = Dataset.get('X')[()]
        Y = Dataset.get('Y')[()]
        # pca=PCA(n_components=0.9,random_state=4)
        # X=pca.fit(X).transform(X)
        Y=np.where(Y==1)
        Y=np.transpose(Y)
        Y=np.delete(Y,0,1)
        Y=np.reshape(Y,(len(Y),))
        return X,Y

# making datasets with output as a single class
    def classDivision(self,X,Y):
        self.classes=np.unique(Y)
        indexes_of_classes={}
        clas,count=np.unique(Y,return_counts=True)
        self.freq_classes=dict(zip(clas,count))
        for i in self.classes:
            a=np.argwhere(Y==i)
            indexes_of_classes[i]=np.reshape(a,(len(a),))  #Indexes where label==(i)
            self.subData[i]=X[indexes_of_classes[i],:]   #each key has a value equal to indexes where label=key
            self.freq_classes[i]/=len(Y) #percentage of all classes           
    # calulating mean and std dev
    def fit(self,X,Y):
        X_new=self.subData
        for i in self.classes:
            self.mean_classes[i]=np.mean(X_new[i],axis=0) #mean of every column for class=i
            self.standard_dev[i]=np.std(X_new[i],axis=0) #STD of every column for class=i

    def gauss(self,X,mean,standard_dev):
        p=(1/(math.sqrt(2*math.pi)*standard_dev))*(math.exp(-(((X-mean)**2)/(2*standard_dev**2))))
        if (p>0):
            return p
        else:
            return 1

# calculating probabilities for each class as output
    def probabs(self,X):
        class_probab={}
        for i in self.classes:
            class_probab[i]=math.log(self.freq_classes[i],math.e) #p(y)
        for i in self.classes:    #calc probability for each output and then we'll take max for prediction
            for j in range(len(X)):
                class_probab[i]+=math.log(self.gauss(X[j],self.mean_classes[i][j],self.standard_dev[i][j]),math.e)
        return class_probab

# predicting by taking max probability
    def predict(self,X):
        prediction=[]
        for i in X:
            probabilities=self.probabs(i)
            maxi=float('-inf')
            output=None
            for clas,p in probabilities.items():
                if p>maxi:
                    maxi=p
                    output=clas
            prediction.append(output)
        return prediction

def accuracy(Y_pred,Y):
    correct=0
    for i in range(len(Y)):
        if (Y_pred[i]==Y[i]):
            correct+=1
    print(correct/len(Y))

def scratch_solve(Dataset):
    nb=NaiveBayes()
    X,Y=nb.preProcess(Dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=4, train_size = 0.8)
    nb.classDivision(X_train,Y_train)
    nb.fit(X,Y)
    ans=nb.predict(X_test)
    accuracy(ans,Y_test)

def sk_solve(Dataset):
    nb1=NaiveBayes()
    X,Y=nb1.preProcess(Dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=4, train_size = 0.8)
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    ans=clf.predict(X_test)
    accuracy(ans,Y_test)


DatasetA=h5.File("part_A_train.h5",'r')
DatasetB=h5.File("part_B_train.h5","r")


print('Dataset A train-test split 80:20')
scratch_solve(DatasetA)
print('Dataset A train-test split 80:20 SkLearn')
sk_solve(DatasetA)


print('Dataset B train-test split 80:20')
scratch_solve(DatasetB)
print('Dataset B train-test split 80:20 SkLearn')
sk_solve(DatasetB)