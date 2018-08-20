# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 13:40:27 2018

@author: Xiangjie Zhao

These codes aim to compare performance between PCA
and t-SNE on different datasets, and try to tune
the parameters of t-SNE to get to its optimal solution
"""
print(__doc__)

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.datasets import load_iris,load_digits,samples_generator
from sklearn.preprocessing import scale,LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


################################################
# compare1
################################################
# data1
iris = load_iris()
X_ir, y_ir = (iris.data, iris.target)
       
# data2
digits = load_digits()
X_dig, y_dig = (digits.data, digits.target)
 
# data3        
data = pd.read_csv('GSE82187_cast_all_forGEO.csv') 
c_delete = ['Unnamed: 0','cell.name','experiment','protocol']
data = data.drop(c_delete,1)
X_sc, y_sc = (data.iloc[:,1:].values, data.iloc[:,0])
y_sc = LabelEncoder().fit_transform(list(y_sc))

# combine datasets
datasets = [
         [scale(X_ir), y_ir, 'iris'],
         [scale(X_dig), y_dig,'digits'],
         [scale(X_sc), y_sc, 'scRNAseq']
         ]

# method1: PCA
pca = PCA(n_components=2)
# method2: t-SNE
tsne = TSNE(n_components=2,random_state=0)
# combine two methods
methods = [
        ('PCA',pca),
        ('t-SNE',tsne)
        ]

# plot performance of the two methods
fig = plt.figure(figsize=(10,7))
plot_num = 1
for data, y, data_name in datasets:
        for name, algorithm in methods:
            t0 = time.time()
            data_mapped = algorithm.fit_transform(data)
            t1 = time.time()
            ax = fig.add_subplot(len(datasets),len(methods),plot_num)
            if plot_num < 3:
                ax.set(title=name)
            ax.scatter(data_mapped[:,0],data_mapped[:,1],c=y,cmap='Set3',s=10)
            ax.text(.99, .05, '%.2fs' % (t1-t0),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
            if plot_num % 2 == 1:
                ax.set(ylabel='%s' % data_name)
            plot_num += 1
            
plt.savefig('compare1.png',dpi=1200)
plt.show()

##############################################
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder,scale
from sklearn.decomposition import PCA
# data4
meta_data = pd.read_csv('GSE110692_indrops_MouseTreg_VATSpleen_metadata.txt',sep=' ') 
counts_data = pd.read_csv('GSE110692_indrops_MouseTreg_VATSpleen_counts.txt',sep=' ')
cell_type = meta_data['cell_type']
counts_data.columns = cell_type
#counts_data.to_csv('GSE110692_indrops_MouseTreg_VATSpleen_dataCleared.csv')

data = counts_data.transpose()
X_sc, y_sc = (data.values, data.index)
y_sc = LabelEncoder().fit_transform(y_sc)
X_sc = scale(X_sc)

tsne = TSNE(n_components=2,init='pca',random_state=1).fit_transform(X_sc)
plt.scatter(tsne[:,0],tsne[:,1],c=y_sc,cmap='Set2')
plt.savefig('data4-2.png',dpi=600)
plt.show()

pca = PCA(n_components=2,random_state=1).fit(X_sc)
var = pca.explained_variance_ratio_*100
pca_ = pca.transform(X_sc)
plt.scatter(pca_[:,0],pca_[:,1],c=y_sc,cmap='Set2')
plt.xlabel('PC1 %.2f' % var[0])
plt.xlabel('PC2 %.2f' % var[1])
plt.savefig('data4-pca2.png',dpi=600)
plt.show()
################################################
# compare2
################################################
# Next line to silence pyflakes. This import is needed.
Axes3D
n_points = 1000
n_components = 2
X, color = samples_generator.make_s_curve(n_points,random_state=0)

fig = plt.figure(figsize=(10,4))
fig.subplots_adjust(wspace=.5)
# 3D origin
ax = fig.add_subplot(131,projection='3d')
ax.view_init(5,-55)
ax.scatter(X[:, 0],X[:, 1],X[:, 2],c=color,cmap='Spectral')

# PCA
pca = PCA(n_components,random_state=1)
t0 = time.time()
X_pca = pca.fit_transform(X)
t1 = time.time()
var = 100*pca.explained_variance_ratio_
ax = fig.add_subplot(132)
ax.scatter(X_pca[:,0],X_pca[:,1],c=color,cmap='Spectral')
ax.set(xlabel='PC1  var: %.2f%%' % var[0], ylabel='PC2  var: %.2f%%' % var[1])
ax.text(.99, .90, '%.2fs' % (t1-t0),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
# TNSE
tsne = TSNE(n_components,init='pca',random_state=1)
tsne_fit = tsne.fit(X)
t0 = time.time()
X_tsne = tsne.fit_transform(X)
t1 = time.time()
ax = fig.add_subplot(133)
ax.scatter(X_tsne[:,0],X_tsne[:,1],c=color,cmap='Spectral')
ax.set(xlabel='TSNE1',ylabel='TSNE2')
ax.text(.99, .90, '%.2fs' % (t1-t0),
                 transform=plt.gca().transAxes,size=10,
                 horizontalalignment='right')

plt.savefig('compare2.png',dpi=1200)
plt.show()


################################################
# compare3
################################################
# t-SNE: tune hyperparameters -- perplexity
perplexity = [2,30,100,200,500]
fig = plt.figure(figsize=(15,10))
plot_num = 1
for perp in perplexity:
    tsne = TSNE(n_components=2,perplexity=perp,random_state=1,init='pca')
    t0 = time.time()
    X_tsne = tsne.fit_transform(X_dig)
    t1 = time.time()
    ax = fig.add_subplot(2,3,plot_num)
    ax.scatter(X_tsne[:,0],X_tsne[:,1],c=y_dig,cmap='Set3')
    ax.text(.99, .05, 'perplexity: %d\ntime: %.2fs' % (perp,(t1-t0)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
    plot_num += 1
plt.savefig('tune1-1.png',dpi=600)
plt.show()

# t-SNE: tune hyperparameters -- learning rate
learn_rate = [2,50,100,200,500,1000]
fig = plt.figure(figsize=(15,10))
plot_num = 1
for lr in learn_rate:
    tsne = TSNE(n_components=2,perplexity=30,learning_rate=lr,random_state=1,init='pca')
    t0 = time.time()
    X_tsne = tsne.fit_transform(X_dig)
    t1 = time.time()
    ax = fig.add_subplot(2,3,plot_num)
    ax.scatter(X_tsne[:,0],X_tsne[:,1],c=y_dig,cmap='Set3')
    ax.text(.99, .05, 'learn_rate: %d\ntime: %.2fs' % (lr,(t1-t0)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
    plot_num += 1
plt.savefig('tune2.png',dpi=600)
plt.show()

# t-SNE: tune hyperparameters -- number of iterative steps
iters = [2,10,100,200,500,1000]
fig = plt.figure(figsize=(15,10))
plot_num = 1
for i in iters:
    tsne = TSNE(n_components=2,random_state=1,init='pca',n_iter_without_progress=i)
    t0 = time.time()
    X_tsne = tsne.fit_transform(X_dig)
    t1 = time.time()
    ax = fig.add_subplot(2,3,plot_num)
    ax.scatter(X_tsne[:,0],X_tsne[:,1],c=y_dig,cmap='Set3')
    ax.text(.99, .05, 'Step: %d\ntime: %.2fs' % (i,(t1-t0)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
    plot_num += 1
plt.savefig('tune3.png',dpi=600)
plt.show()
