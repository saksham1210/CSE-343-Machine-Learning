import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Pre-processing Data
def preProcess():
    DatasetA=h5.File("part_A_train.h5",'r')
    X = DatasetA.get('X')[()]
    Y=DatasetA.get('Y')[()]
    Y=np.where(Y==1)
    Y=np.transpose(Y)
    Y=np.delete(Y,0,1)
    Y=np.reshape(Y,(4200,))
    return X,Y

#sampling the data 
def Stratifiedsampling(X,Y):
    # Stratified Sampling
    sss = StratifiedShuffleSplit(test_size=0.2,train_size=0.8, random_state=0)
    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
    return X_train,X_test,Y_train,Y_test

#calculates class frequencies
def classfreq(Y_train,Y_test):
    unique_test,count_test=np.unique(Y_test,return_counts=True)
    count_test=(count_test/len(Y_test))*100
    print("Test data Class freq")
    print("[class,percentage]")
    print(np.asarray((unique_test,count_test)).T)
    unique_train,count_train=np.unique(Y_train,return_counts=True)
    count_train=(count_train/len(Y_train))*100
    print("--------------------------------------------------")
    print("Train data Class freq")
    print("[class,percentage]")
    print(np.asarray((unique_train,count_train)).T)

# Logistic regression
def logisticAccu(X_train,X_test,Y_train,Y_test,part):
    # Logistic
    logistic=LogisticRegression(max_iter=10000)
    logistic.fit(X_train,Y_train)
    acc=logistic.score(X_test,Y_test)
    if (part=='pca'):
        print("PCA+Logistic Acuu = ",acc)
    else:
        print("SVD+Logistic Acuu = ",acc)

def pca(X):
	scaler=StandardScaler()
	X1=scaler.fit(X).transform(X)
	pca=PCA(n_components=200,random_state=0)
	X_pca=pca.fit(X1).transform(X1)
	return X_pca

def svd(X):
    svd = TruncatedSVD(n_components=200,random_state=0)
    X_svd=svd.fit(X).transform(X)
    return X_svd

def tsne(X,Y):
    # TSNE
    tsn=TSNE(random_state=0)
    X_sn=tsn.fit_transform(X)
    sns.scatterplot(X_sn[:,0], X_sn[:,1], hue=np.transpose(Y), legend='full',palette=sns.color_palette("hls", 10))
    plt.show()


X,Y=preProcess()
print()
print("------------Stratified Sampling------------")
print()
X_train,X_test,Y_train,Y_test=Stratifiedsampling(X,Y)
classfreq(Y_train,Y_test)


X_pca=pca(X)
print()
print("----------PCA-----------")
print()
X_train_pca,X_test_pca,Y_train_pca,Y_test_pca=Stratifiedsampling(X_pca,Y)
logisticAccu(X_train_pca,X_test_pca,Y_train_pca,Y_test_pca,'pca')
tsne(X_train_pca,Y_train_pca)


X_svd=svd(X)
print()
print("------------SVD--------------")
print()
X_train_svd,X_test_svd,Y_train_svd,Y_test_svd=Stratifiedsampling(X_svd,Y)
logisticAccu(X_train_svd,X_test_svd,Y_train_svd,Y_test_svd,'svd')
tsne(X_train_svd,Y_train_svd)
















