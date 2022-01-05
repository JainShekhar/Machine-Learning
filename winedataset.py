# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 12:40:20 2021

@author: shejain
"""
#showing PCA, LDA, KPCA with wine dataset
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
#note eigh returns eigvals and eigvecs in ascending order of eigenvals but only works for symmetric matrices

wine = datasets.load_wine()
#print(wine.DESCR)
df = pd.DataFrame(wine.data, columns = wine.feature_names)
y = wine.target
#print(wine.target_names)
X = np.array(df)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, stratify = y, random_state = 0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

####################################################################
#implementing PCA
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
var_exp = [(i/eigen_vals.sum()) for i in eigen_vals]
plt.bar(range(1,14),var_exp,alpha=1,align='center',label='Individual explained variance')
plt.ylabel('Explained variance')
plt.xlabel('Principal component index')
plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse = True)
w = np.hstack((eigen_pairs[0][1][:,np.newaxis],
               eigen_pairs[1][1][:,np.newaxis]))
X_train_std_pca = X_train_std.dot(w)
plt.figure()
colors = ['r','m','g']
markers = ['s','x','o']
for idx, c in enumerate(np.unique(y_train)):
    plt.scatter(X_train_std_pca[y_train==c,0],X_train_std_pca[y_train==c,1],color=colors[idx],label=c,marker=markers[idx])
    
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(loc='lower left')
plt.show()

#########################################################################
#implementing LDA
#calculate mean vectors
mean_vecs = []
for label in range(3):
    mean_vecs.append(np.mean(X_train_std[y_train == label],axis=0))
#calculate the inidividual scatter matrices and sum to get within-class scatter matrix (Sw)
d = 13
S_W = np.zeros((d,d))
for label in range(3):
    individual_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += individual_scatter

#calculate the between class scatter matrix
mean_overall = np.mean(X_train_std, axis=0)
om = mean_overall.reshape(d,1)
S_B = np.zeros((d,d))
for label, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == label, :].shape[0]
    mv = mean_vec.reshape(d,1)
    S_B += n*(mv - om).dot((mv - om).T)

#now find eigenvalues and eigenvectors of inv(S_W).dot(S_B)
eigen_vals_LDA, eigen_vecs_LDA = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

#sort them
eigen_pairs_LDA = [(eigen_vals_LDA[i],eigen_vecs_LDA[:,i]) for i in range(len(eigen_vals_LDA))]
eigen_pairs.sort(key=lambda k: k[0], reverse = True)

#make transformation matrix with top two eigen vectors
w = np.column_stack((eigen_pairs[0][1],eigen_pairs[1][1]))
#trasform X_train_std
X_train_std_lda = X_train_std.dot(w)
plt.figure()
colors = ['r','m','g']
markers = ['s','x','o']
for idx, c in enumerate(np.unique(y_train)):
    plt.scatter(X_train_std_lda[y_train==c,0],X_train_std_lda[y_train==c,1],color=colors[idx],label=c,marker=markers[idx])
    
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.legend(loc='lower left')
plt.show()

##################################################################33
#implementing KPCA function
def rbf_kernel_PCA (X,gamma,n_components):
    #inputs: X matrix (n,d), gamma is the RBF kernel parameter, n_components are numer of principal components to return
    #output: X_kpca which is the projected dataset, alphas, lambdas are the top n_components eigenvectors and eigenvalues of centered kernel matrix
    
    #calculate pairwise squared-Euclidean distances
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    
    #make Kernel matrix
    K = exp(-gamma*mat_sq_dists)
    
    #center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N))/N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    #now obtain eigen pairs in ascending order of eigenvals, note eigh works only for symmetric matrics ow use eig
    eigenvals, eigenvecs = np.linalg.eigh(K)
    #flip the order to get in descending order
    eigenvals, eigenvecs = eigenvals[::-1], eigenvecs[:,::-1] 
    
    #collect top n_components eigenvectors and eigenvals
    alphas = np.column_stack([eigenvecs[:,i] for i in range(n_components)])
    lambdas = [eigenvals[i] for i in range(n_components)]
    
    X_kpca = alphas
    return(X_kpca, alphas, lambdas)

#define function to project a new dataset on KPCA axis
def project_x(x_new, x, gamma, alphas, lambdas):
    #input: x_new is new dataset, x is original dataset, gamma is kernel function para, alphas, lambdas are eigenvecs and eigenvals
    #output: x_proj is projected x_new
    pair_dist = np.array([np.sum(x_new - row)**2 for row in x])
    k = np.exp(-gamma*pair_dist)
    x_proj = k.dot(alphas/lambdas)
    return x_proj
    
#implementing KPCA for moon_dataset
#first make binary classification datasets with half moons
x1, y1 = make_moons(n_samples = 100, random_state = 123)
#now apply KPCA
gamma = 15
x_kpca, alphas, lambdas = rbf_kernel_PCA(x1,gamma=gamma,n_components=2)
#try projecting a new dataset
x_new = np.array([0,0.4])
x_proj = project_x(x_new, x1, gamma, alphas, lambdas)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(7,3))
ax[0].scatter(x1[y1==0,0],x1[y1==0,1],color='red',marker='^',alpha=0.5,label='0')
ax[0].scatter(x1[y1==1,0],x1[y1==1,1],color='blue',marker='o',alpha=0.5,label='1')
ax[0].scatter(x_new[0],x_new[1],color='black',marker='x',alpha=0.5,s = 100, label='new dataset')
ax[1].scatter(x_kpca[y1==0,0],x_kpca[y1==0,1],color='red',marker='^',alpha=0.5,label='0')
ax[1].scatter(x_kpca[y1==1,0],x_kpca[y1==1,1],color='blue',marker='o',alpha=0.5,label='0')
ax[1].scatter(x_proj[0],x_proj[1],color='black',marker='x',alpha=0.5,s = 100, label='projected new dataset')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[0].legend(loc='upper right')
ax[1].set_xlabel('PCA1')
ax[1].set_ylabel('PCA2')
ax[1].legend(loc='upper right')
plt.show()      




