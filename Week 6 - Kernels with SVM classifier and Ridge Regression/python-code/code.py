#!/usr/bin/env python
# coding: utf-8

# CS7CS4/CSU44061 Machine Learning
# Week 6 Assignment
# Boris Flesch (20300025)
# 
# Downloaded dataset
# id:23-69-115

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def plotData(X, y):
    plt.scatter(X, y, c='r', marker='+', label="Training data")
    plt.gca().set(title="Training data visualisation", xlabel="X", ylabel="y")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()


# (i)
X = np.array([-1, 0, 1]).reshape(-1, 1)
y = np.array([0, 1, 0])
plotData(X,y)


# (i)(a)
def gaussian_kernel0(distances):
    weights = np.exp(0*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel1(distances):
    weights = np.exp(-1*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel5(distances):
    weights = np.exp(-5*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel10(distances):
    weights = np.exp(-10*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel25(distances):
    weights = np.exp(-25*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel50(distances):
    weights = np.exp(-50*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel100(distances):
    weights = np.exp(-100*(distances**2))
    return weights/np.sum(weights)


def knnGaussianKernel(X, y, k, gaussianKernels):
    plt.figure(num=None, figsize=(8, 5), dpi=120)
    plt.rc('font', size=12)
    
    if (X.size < 10):
        plt.scatter(X, y, color='black', marker='+', s=10, linewidth=15, label="train")
    else:
        plt.scatter(X, y, color='black', marker='+', label="train")
    
    Xtest = np.linspace(-3.0, 3.0, num=1000).reshape(-1, 1)
    
    for gaussianKernel in gaussianKernels:
        model = KNeighborsRegressor(n_neighbors=k, weights=gaussianKernel[1]).fit(X, y)
        ypred = model.predict(Xtest)
        
        plt.plot(Xtest, ypred, label="Predictions - γ=%d"%gaussianKernel[0])
    
    plt.xlabel("X"); plt.ylabel("y")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title("kNN using Gaussian weights - k = %d" % k)
    plt.show()

kernels = [
    [0, gaussian_kernel0],
    [1, gaussian_kernel1],
    [5, gaussian_kernel5],
    [10, gaussian_kernel10],
    [25, gaussian_kernel25]
]

knnGaussianKernel(X, y, k=3, gaussianKernels=kernels)


# (i)(c)
def kernalisedRidgeRegression(X, y, C, gaussianKernels, printCoeff=False):
    plt.figure(num=None, figsize=(8, 5), dpi=120)
    plt.rc('font', size=12); plt.rcParams['figure.constrained_layout.use'] = True
    
    if (X.size < 10):
        plt.scatter(X, y, color='black', marker='+', s=10, linewidth=15, label="train")
    else:
        plt.scatter(X, y, color='black', marker='+', label="train")
    
    for gaussianKernel in gaussianKernels:
        model = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=gaussianKernel[0]).fit(X, y)
        Xtest = np.linspace(-3.0, 3.0, num=1000).reshape(-1, 1)
        ypred = model.predict(Xtest)
        plt.plot(Xtest, ypred, label="Predictions - γ=%d"%gaussianKernel[0])
        if (printCoeff):
            print("Kernel Ridge Regression - C = %d, γ=%d" % (C, gaussianKernel[0]))
            print("θ =", model.dual_coef_)
            print("--")
        
    plt.xlabel("X"); plt.ylabel("y")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title("Kernel Ridge Regression - C = %.1f" % (C))
    plt.show()

kernalisedRidgeRegression(X, y, C=0.1, gaussianKernels=kernels, printCoeff=True)
kernalisedRidgeRegression(X, y, C=1, gaussianKernels=kernels, printCoeff=True)
kernalisedRidgeRegression(X, y, C=100, gaussianKernels=kernels, printCoeff=True)


# (ii)
def readData(filepath):
    df = pd.read_csv(filepath, comment="#")
    X = df.iloc[:,0]
    y = df.iloc[:,1]
    return np.array(X).reshape(-1, 1), np.array(y)

X,y = readData("week6.csv")
plotData(X,y)


# (ii)(a)
kernels = [
    [0, gaussian_kernel0],
    [1, gaussian_kernel1],
    [5, gaussian_kernel5],
    [10, gaussian_kernel10],
    [25, gaussian_kernel25]
]

knnGaussianKernel(X, y, k=X.size, gaussianKernels=kernels)

print("Output y average: %.3f" % y.mean())

Xm1_y = []
Xp1_y = []
for i in range(X.size):
    if X[i] == -1:
        Xm1_y.append(y[i])
    elif X[i] == 1:
        Xp1_y.append(y[i])

print("Output y average for training data points where x=-1: %.3f"%np.mean(Xm1_y))
print("Output y average for training data points where x=1: %.3f"%np.mean(Xp1_y))


# (ii)(b)
C_range = [
    # 0.1,
    1,
    10,
    50,
    100,
    # 500,
    # 1000
]
kernels = [
    #[0, gaussian_kernel0],
    [1, gaussian_kernel1],
    [5, gaussian_kernel5],
    [10, gaussian_kernel10],
    [25, gaussian_kernel25]
]

for gaussianKernel in kernels:
    mean_error = []
    std_error = []
    for C in C_range:
        model = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=gaussianKernel[0])
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_error.append(-np.array(scores).mean())
        std_error.append(np.array(scores).std())

    plt.errorbar(C_range, mean_error, yerr=std_error, linewidth=3, label="γ = %d"%gaussianKernel[0])

plt.title("Error mean and variance for C cross-validation")
plt.gca().set(xlabel='C penalty', ylabel='Mean square error')
plt.legend()
plt.show()

kernels = [
    [0, gaussian_kernel0],
    [1, gaussian_kernel1],
    [5, gaussian_kernel5],
    [10, gaussian_kernel10],
    [25, gaussian_kernel25]
]

kernalisedRidgeRegression(X, y, C=100, gaussianKernels=kernels)


# (ii)(c)
std_error = []
mean_error = []
tested_kernels = []
kernels = [
    # [0, gaussian_kernel0],
    # [1, gaussian_kernel1],
    [5, gaussian_kernel5],
    [10, gaussian_kernel10],
    [25, gaussian_kernel25],
    [50, gaussian_kernel50],
    [100, gaussian_kernel100]
]

for gaussianKernel in kernels:
    
    kf = KFold(n_splits=5)
    temp = []
    for train, test in kf.split(X):
        model = KNeighborsRegressor(n_neighbors=X[train].size, weights=gaussianKernel[1])
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test], ypred))

    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
    tested_kernels.append(gaussianKernel[0])

plt.errorbar(tested_kernels, mean_error, yerr=std_error, linewidth=3)
plt.title("Error mean and variance for γ cross-validation in kNN")
plt.gca().set(xlabel='Value of γ', ylabel='Mean square error')
plt.show()


C = 10

std_error = []
mean_error = []
tested_kernels = []

kernels = [
    # [0, gaussian_kernel0],
    [1, gaussian_kernel1],
    [5, gaussian_kernel5],
    [10, gaussian_kernel10],
    [25, gaussian_kernel25],
    [50, gaussian_kernel50],
    [100, gaussian_kernel100],
    # [500, gaussian_kernel500]
]

for gaussianKernel in kernels:
    model = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=gaussianKernel[0])
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mean_error.append(-np.array(scores).mean())
    std_error.append(np.array(scores).std())
    tested_kernels.append(gaussianKernel[0])

plt.errorbar(tested_kernels, mean_error, yerr=std_error, linewidth=3)
plt.title("Error mean and variance for γ cross-validation in KRR")
plt.gca().set(xlabel='Value of γ', ylabel='Mean square error')
plt.show()


knnGaussianKernel(X, y, k=X.size, gaussianKernels=[[50, gaussian_kernel50]])
kernalisedRidgeRegression(X, y, C=10, gaussianKernels=[[10, gaussian_kernel10]])
