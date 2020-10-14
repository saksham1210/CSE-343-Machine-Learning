import h5py as h5
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA


class Q3():
    def __init__(self):
        self.model=None
        self.model_GNB=None
        self.X_test=None
        self.Y_test=None
        self.best_mean_DT_accu=0
        self.best_mean_GNB_accu=0
        self.Y_pred=None
        self.filename=None

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

    #Return VAL_TRAIN Splitted Arrays
    def split_return(self,i,folds_x,folds_y):
        Xt=folds_x.copy()
        X_val=np.array(folds_x[i],dtype=object)
        del Xt[i]
        Xt=np.array(Xt,dtype=object)
        l=len(Xt)
        Xt_final=Xt[0]
        for j in range(1,l):
            Xt_final=np.concatenate((Xt_final,Xt[j]))
        Yt=folds_y.copy()
        Y_val=np.array(folds_y[i],dtype=object)
        del Yt[i]
        Yt=np.array(Yt,dtype=object)
        l=len(Yt)
        Yt_final=Yt[0]
        for j in range(1,l):
            Yt_final=np.concatenate((Yt_final,Yt[j])) 
        Yt_final=Yt_final.astype('int')
        Xt_final=Xt_final.astype('int')
        X_val=X_val.astype('int')
        Y_val=Y_val.astype('int')
        return Xt_final,X_val,Yt_final,Y_val

    # Decision tree classifier, return accuracy and model
    def DesTree(self,depth,X_train,X_val,Y_train,Y_val):
        clf = DecisionTreeClassifier(max_depth=depth,random_state=4)
        clf.fit(X_train,Y_train)
        Y_pred=clf.predict(X_val)
        Y_pred_train=clf.predict(X_train)
        accu=self.accuracy(Y_pred,Y_val)
        train_accu=self.accuracy(Y_pred_train,Y_train)
        return accu,train_accu,clf
     # GNB classifier, return accuracy and model   
    def GNB(self,depth,X_train,X_val,Y_train,Y_val):
        clf = GaussianNB()
        clf.fit(X_train, Y_train)
        ans=clf.predict(X_val)
        accu=self.accuracy(ans,Y_val)
        return accu,clf

    # K-Fold CV, returns accuracy and best model
    def Kfold(self,X,Y,k,max_depth,model):
        accus=[]
        if model=='DT':
            train_accus=[]
        models=[]
        X_train_val, self.X_test, Y_train_val, self.Y_test = train_test_split(X, Y, random_state=4, train_size = 0.8) #split into train+val and test
        folds_x=np.array_split(X_train_val,k)
        folds_y=np.array_split(Y_train_val,k)
        # traversing in all folds and making val and trains
        for i in range(k):
            X_train,X_val,Y_train,Y_val=self.split_return(i,folds_x,folds_y)
            if model=="DT":
                accu,train_accu,clf=self.DesTree(max_depth,X_train,X_val,Y_train,Y_val)
            else:
                accu,clf=self.GNB(max_depth,X_train,X_val,Y_train,Y_val)
            models.append(clf)
            accus.append(accu)
            if model=="DT":
                train_accus.append(train_accu)
        best_model=models[accus.index(max(accus))]
        if model=="DT":
            return sum(accus)/k , sum(train_accus)/k , best_model
        else:
            return sum(accus)/k , best_model


    def gridsearch(self,X,Y,max_depth):
        mean_accu=[]
        train_mean_accu=[]
        models=[]   #best model for each depth
        for i in range(1,max_depth):
            accu,train_accu,model=self.Kfold(X,Y,4,i,"DT")
            mean_accu.append(accu)
            train_mean_accu.append(train_accu)
            models.append(model)
        depth=mean_accu.index(max(mean_accu))+1
        print("Max mean Validation accuracy at depth=",depth)
        depths=[i for i in range(1,max_depth)]
        plt.plot(depths,mean_accu,label="Validation Accuracy",color='b')
        plt.plot(depths,train_mean_accu,label='Training Accuracy',color='r')
        plt.xlabel("Tree Depth")
        plt.ylabel("Values")
        plt.show()
        return depth,models[depth-1],max(mean_accu)

    def fit(self,Dataset,name):
        X,Y=self.preProcess(Dataset)
        depth,self.model,self.best_mean_DT_accu=self.gridsearch(X,Y,50) #DT
        self.best_mean_GNB_accu,self.model_GNB=self.Kfold(X,Y,4,0,'GNB')
        print("Best mean accu using GNB",self.best_mean_GNB_accu)
        print("Best mean accu using DT",self.best_mean_DT_accu)
        if self.best_mean_DT_accu>=self.best_mean_GNB_accu:
            self.saveModel(self.model,name)
        else:
            self.saveModel(self.model_GNB,name)
        

    def saveModel(self,model,Dataset):
        filename = 'finalized_model'+Dataset+'.sav'
        self.filename=filename
        pickle.dump(model, open(filename, 'wb'))

    def loadModel(self):
        loaded_model = pickle.load(open(self.filename, 'rb'))
        return loaded_model

    def predict_test(self):
        model=self.loadModel()
        self.Y_pred=model.predict(self.X_test)
        if self.best_mean_DT_accu>=self.best_mean_GNB_accu:
            print("Predicting using best DT model")
        else:
            print("Predicting using best GNB model")
        print("Testing Accuracy = ",self.accuracy(self.Y_pred,self.Y_test))

    def accuracy(self,Y_pred,Y):
        correct=0
        for i in range(len(Y)):
            if (Y_pred[i]==Y[i]):
                correct+=1
        return correct/len(Y)

    def microMacro(self,Y_pred,Y_test):
        micro_precision_num=0
        micro_precision_den=0
        micro_recall_num=0
        micro_recall_den=0
        recall_sum=0
        precision_sum=0
        for clas in range(10):
            tp=0
            tn=0
            fp=0
            fn=0
            for i in range(len(Y_pred)):
                if (Y_pred[i]==clas): 
                    if (Y_test[i]==clas):
                        tp+=1
                    else:
                        fp+=1
                if (Y_pred[i]!=clas): 
                    if (Y_test[i]!=clas):
                        tn+=1
                    else:
                        fn+=1
            micro_precision_num+=tp
            micro_precision_den+=(fp+tp)
            precision_sum+=(tp/(tp+fp))
            recall_sum+=(tp/(tp+fn))
            micro_recall_num+=tp
            micro_recall_den+=(tp+fn)
        print("Micro Recall = ",(micro_recall_num/micro_recall_den))
        print("Micro Precision = ",(micro_precision_num/micro_precision_den))
        print("Macro Recall = ",(recall_sum/10))
        print("Macro Precision = ",(precision_sum/10))

    def confusionMatrix(self,Y_pred,Y_test,clas):
        tp=0
        tn=0
        fp=0
        fn=0
        for i in range(len(Y_pred)):
            if (Y_pred[i]==clas): 
                if (Y_test[i]==clas):
                    tp+=1
                else:
                    fp+=1
            if (Y_pred[i]!=clas): 
                if (Y_test[i]!=clas):
                    tn+=1
                else:
                    fn+=1
        return tp,tn,fp,fn

    def precision(self,Y_pred,Y_test,clas):
        tp,tn,fp,fn=self.confusionMatrix(Y_pred,Y_test,clas)
        denominator=tp+fp
        return (tp/denominator)

    def recall(self,Y_pred,Y_test,clas):
        tp,tn,fp,fn=self.confusionMatrix(Y_pred,Y_test,clas)
        denominator=tp+fn
        return (tp/denominator)

    def f1Score(self,Y_pred,Y_test,clas):
        p=self.precision(Y_pred,Y_test,clas)
        r=self.recall(Y_pred,Y_test,clas)
        num=p*r*2
        denom=p+r
        return num/denom

    def tpr(self,Y_pred,Y_test,clas):
        tp,tn,fp,fn=self.confusionMatrix(Y_pred,Y_test,clas)
        return (tp/(tp+fn))

    def fpr(self,Y_pred,Y_test,clas):
        tp,tn,fp,fn=self.confusionMatrix(Y_pred,Y_test,clas)
        return (fp/(fp+tn))

    def getProbabilities(self):
        return self.model.predict_proba(self.X_test)
        
# predicts new Y with the new threshold
    def getNewYPRED(self,threshold,clas):
        Y_pred=[]
        prob_arr=self.getProbabilities()
        for i in range(len(prob_arr)):
            if (prob_arr[i,clas]>=threshold):
                Y_pred.append(clas)
            else:
                Y_pred.append(-1)
        return Y_pred

    def ROC(self,stepsize,dataset):
        if dataset=="B":
            fpr_x=[]
            tpr_y=[]
            arr=np.arange(0.0, 1.0, stepsize)
            clas=1
            for threshold in arr:
                Y_pred=self.getNewYPRED(threshold,clas)
                tpr_y.append(self.tpr(Y_pred,self.Y_test,clas))
                fpr_x.append(self.fpr(Y_pred,self.Y_test,clas))
            plt.plot(fpr_x,tpr_y)
            plt.plot(fpr_x,fpr_x,linestyle='dashed',color='black')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.show()
        else:
            for i in range(10):
                arr=np.arange(0.0, 1, stepsize)
                fpr_x=[]
                tpr_y=[]
                for threshold in arr:
                    Y_pred=self.getNewYPRED(threshold,i)
                    tpr_y.append(self.tpr(Y_pred,self.Y_test,i))
                    fpr_x.append(self.fpr(Y_pred,self.Y_test,i))
                fpr_x.append(0)
                tpr_y.append(0)
                plt.plot(fpr_x,tpr_y,label=str(i))
            plt.plot(fpr_x,fpr_x,linestyle='dashed',color='black')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.show()

        
    

    def metrics(self,dataset):
        if (dataset=="B"):
            print("Precision=",self.precision(self.Y_pred,self.Y_test,1))
            print('Recall=',self.recall(self.Y_pred,self.Y_test,1))
            print("F1 Score=",self.f1Score(self.Y_pred,self.Y_test,1))
            self.ROC(0.002,dataset)
        else:
            self.ROC(0.002,dataset)
            self.microMacro(self.Y_pred,self.Y_test)

DatasetA=h5.File("part_A_train.h5",'r')
DatasetB=h5.File("part_B_train.h5","r")

print("-------------------------DATASET-A----------------------")
A=Q3()
print("Fitting on datset-A")
A.fit(DatasetA,"A")
print("")
print("Predicting on dataset-A")
print("")
A.predict_test()
A.metrics("A")


print("-------------------------DATASET-B----------------------")
B=Q3()
print("Fitting on datset-B")
B.fit(DatasetB,"B")
print("")
print("Predicting on dataset-B")
print("")
B.predict_test()
B.metrics("B")

