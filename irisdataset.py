
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:36:56 2021

@author: shejain
"""
#all classifiers with iris dataset
import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.pipeline import _name_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#first define Perceptron classifier with binary classification: 0 and 1
class Perceptron1:
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = np.empty_like(y)
            output = np.empty_like(y)
            misclassify = 0
            for i in range(len(y)):
                #predict with current weights
                output[i] = self.predict(X[i])
                #now calculate errors and updates the weights
                errors[i] = (y[i] - output[i])               
                self.w_[1:] += self.eta*errors[i]*X[i]
                self.w_[0] +=self.eta*errors[i]
                #count number of misclassifications          
                misclassify += int(errors[i]!=0)         
            self.errors_.append(misclassify)
        return self
       
    def predict(self,x):
        return np.where((np.dot(x,self.w_[1:]) + self.w_[0])>= 0.0, 1, -1)

class AdalineGD:
    def __init__(self,eta=0.001,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            #calculate error
            ninput = self.net_input(X)
            nactivation = self.activation(ninput)
            errors = (y - nactivation)
            
            #update weights now
            self.w_[1:] += self.eta*(X.T.dot(errors))
            self.w_[0] += self.eta*(errors.sum())
            
            #calculate cost
            cost = 0.5*(errors**2).sum()                               
            self.cost_.append(cost)           
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def activation(self,X):
        return X
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>= 0.0, 1, -1)

iris = datasets.load_iris()

df = pd.DataFrame(iris.data)
#only consider the two features: sepal length and sepal width
#only two target values that are the first 100 - setosa and versicolor
X = np.array(df[[0,2]][:100])
#same as X = np.array(df.iloc[0:100,0:2].values)
y = np.array(iris.target[:100])
for i in range(len(y)):
    y[i] = np.where(y[i] == 0, -1, 1)
    
ppn1 = AdalineGD(eta=0.01,n_iter=15)
ppn1.fit(X,y)
#plt.plot(range(1,len(ppn.cost_)+1),ppn.cost_,label='non-standardized')

#now standardize variables
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()
ppn1.fit(X_std,y)
#plt.plot(range(1,len(ppn1.cost_)+1),ppn1.cost_,label='standardized')
#plt.xlabel('number of iterations')
#plt.ylabel('cost function')
#plt.legend()
#plt.show()

X1 = iris.data[:,[2,3]]
y1 = iris.target

#split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=1, stratify=y1)
#standardize
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1, max_iter=1000)
ppn.fit(X_train_std,y_train)
y_pred = ppn.predict(X_test_std)
#print('misclassified are %d' %(y_test != y_pred).sum())

def plot_decision_regions(X,y,classifier,resolution=0.02):
    
    #define colors and markers
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #create points for surface
    x1min, x1max = X[:,0].min() - 1, X[:,0].max() + 1
    x2min, x2max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1min, x1max,resolution),
                           np.arange(x2min,x2max,resolution))
    x_features = np.array([xx1.ravel(),xx2.ravel()]).T
    y_pred_x_features = classifier.predict(x_features)
    yy2 = y_pred_x_features.reshape(xx1.shape)
    
    #plot the surface
    plt.contourf(xx1,xx2,yy2,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    
    #put the points
    for idx,c1 in enumerate(np.unique(y)):
        if (c1 == 1): 
            string1 = 'setosa'
        elif (c1 == 2):
            string1 = 'versicolor'
        else:
            string1 = 'virginica'
            
        plt.scatter(x=X[y == c1, 0],y=X[y == c1, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=string1, edgecolor='black')
    plt.legend(loc='upper left')


ppn = Perceptron(eta0=0.001,max_iter=500)
ppn.fit(X_train,y_train)
X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X_combined,y_combined,classifier=ppn)
plt.title('Perceptron')

plt.figure()
ppn = LogisticRegression(C = 100,random_state=1,solver='lbfgs',multi_class='ovr')
ppn.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=ppn)
plt.title('LogisticRegression')
nc = {key: value for key, value in _name_estimators(estimators=[ppn])}
print(nc)

plt.figure()
ppn = SVC(kernel='linear',C=1,random_state=1)
ppn.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=ppn)
plt.title('SVM')

plt.figure()
ppn = DecisionTreeClassifier(criterion='gini',max_depth=None,random_state=1)
ppn.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=ppn)
plt.title('DecisionTree')
plt.figure()
tree.plot_tree(ppn)
plt.show()

plt.figure()
ppn = RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1)
ppn.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=ppn)
plt.title('RandomForest')

plt.figure()
ppn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
ppn.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=ppn)
plt.title('KNN')



