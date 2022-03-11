#!/usr/bin/env python
# coding: utf-8

# CS7CS4/CSU44061 Machine Learning
# Week 3 Assignment
# Boris Flesch (20300025)
# 
# Downloaded dataset
# id:22-22-22

# Read data
import numpy as np
import pandas as pd
df = pd.read_csv("week3.csv", comment="#")
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y = df.iloc[:,2]

# (i)(a)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# [[elev, azim], [...]] -> multiple angle of views
figViews = [[10,10], [40,80]]
for figView in figViews:
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], y)
    ax.view_init(elev=figView[0], azim=figView[1])
    ax.set(title='Training data from CSV', xlabel='X1', ylabel='X2', zlabel='Y')
    plt.show()


# (i)(b)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

Xpoly = PolynomialFeatures(5).fit_transform(X)

baseline = DummyRegressor(strategy="mean").fit(Xpoly, y)
print("J(θ_baseline) = %f\n"%mean_squared_error(y, baseline.predict(Xpoly)))

# Prevents scientific notation for theta values
np.set_printoptions(suppress=True)

C_range = [1, 10, 100, 1000]
for Ci in C_range:
    model = Lasso(alpha=1/(2*Ci)).fit(Xpoly, y)
    theta = np.insert(model.coef_, 0, model.intercept_)

    print("C = %.1f"%Ci)
    print("θ =", theta)
    print("J(θ) = %f\n"%mean_squared_error(y, model.predict(Xpoly)))


# (i)(c)
Xtest = []
grid = np.linspace(-5,5)
for i in grid:
    for j in grid:
        Xtest.append([i,j])

Xtest = np.array(Xtest)
Xtest = PolynomialFeatures(5).fit_transform(Xtest)

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

C_range = [1, 10, 100, 1000]
for Ci in C_range:
    model = Lasso(alpha=1/(2*Ci))
    model.fit(Xpoly, y)
    
    fig = plt.figure(num=None, figsize=(8, 5), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:,0], X[:,1], y, c='g', label="Training data")
    surf = ax.plot_trisurf(Xtest[:,1], Xtest[:,2], model.predict(Xtest), cmap=cm.coolwarm, alpha=0.8, linewidth=0, antialiased=True)
    ax.view_init(elev=30, azim=50)
    ax.set_title('Lasso prediction surface for C = %.0f'%Ci)
    ax.set(xlabel='X1', ylabel='X2', zlabel='Y')
    ax.legend(bbox_to_anchor=(0.84, 0.9), loc='upper left')

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Predictions', rotation=270)

    plt.show()


# (i)(e)
from sklearn.linear_model import Ridge

baseline = DummyRegressor(strategy="mean").fit(Xpoly, y)
print("J(θ_baseline) = %f\n"%mean_squared_error(y, baseline.predict(Xpoly)))

C_range = [1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1]
for Ci in C_range:
    model = Ridge(alpha=1/(2*Ci)).fit(Xpoly, y)
    theta = np.insert(model.coef_, 0, model.intercept_)

    print("C = %.0E"%Ci)
    print("θ =", theta)
    print("J(θ) = %f\n"%mean_squared_error(y, model.predict(Xpoly)))
    
    # continue; # Uncomment to display only parameters values (i.e. hide graphs)
    
    fig = plt.figure(num=None, figsize=(8, 5), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:,0], X[:,1], y, c='g', label="Training data")
    surf = ax.plot_trisurf(Xtest[:,1], Xtest[:,2], model.predict(Xtest), cmap=cm.coolwarm, alpha=0.8, linewidth=0, antialiased=True)

    ax.view_init(elev=30, azim=50)
    ax.set_title('Ridge prediction surface for C = %.0E'%Ci)
    ax.set(xlabel='X1', ylabel='X2', zlabel='Y')
    ax.legend(bbox_to_anchor=(0.84, 0.9), loc='upper left')

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Predictions', rotation=270)

    plt.show()


# (ii)(a)
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# k = 5 only
model = Lasso(alpha=1/2) # <=> C = 1
kf = KFold(n_splits=5)
temp = []
for train, test in kf.split(Xpoly):
    model.fit(Xpoly[train], y[train])
    ypred = model.predict(Xpoly[test])
    temp.append(mean_squared_error(y[test], ypred))

print("5-fold cross validation results:")
print("Mean error = %f; Standard deviation = %f"%(np.array(temp).mean(), np.array(temp).std()))

# k-fold cross-validation
plt.figure(num=None, figsize=(8, 6), dpi=120)

model = Lasso(alpha=1/2) # <=> C = 1
k = [2, 5, 10, 25, 50, 100]
std_error = []
mean_error = []

for ki in k:
    kf = KFold(n_splits=ki)
    temp = []
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], y[train])
        ypred = model.predict(Xpoly[test])
        temp.append(mean_squared_error(y[test], ypred))

    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

plt.errorbar(k, mean_error, yerr=std_error, linewidth=3)
plt.title("Error mean and variance for k-fold cross-validation")
plt.gca().set(xlabel='k (number of folds)', ylabel='Mean square error')
plt.show()


# (ii)(b) & (ii)(c)
stdErrorTest, meanErrorTest, stdErrorTrain, meanErrorTrain = [], [], [], []
plt.figure(num=None, figsize=(8, 6), dpi=120)

C_range = [1, 5, 10, 50, 100]
kf = KFold(n_splits=10)

for Ci in C_range:
    model = Lasso(alpha=1/(2*Ci))
    tempTest, tempTrain = [], []
    
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], y[train])
        ypred = model.predict(Xpoly[test])
        ypred_train = model.predict(Xpoly[train])
        
        tempTest.append(mean_squared_error(y[test], ypred))
        tempTrain.append(mean_squared_error(y[train], ypred_train))

    meanErrorTest.append(np.array(tempTest).mean())
    stdErrorTest.append(np.array(tempTest).std())
    meanErrorTrain.append(np.array(tempTrain).mean())
    stdErrorTrain.append(np.array(tempTrain).std())

plt.errorbar(C_range, meanErrorTest, yerr=stdErrorTest, linewidth=3, c="blue", label="Test data")
plt.errorbar(C_range, meanErrorTrain, yerr=stdErrorTrain, linewidth=3, c="orange", label="Training data")
plt.title("Error mean and standard deviation vs C values (Lasso model)")
plt.gca().set(xlabel='C', ylabel="Mean square error")
plt.legend()
plt.show()


#(ii)(d)
C_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

stdErrorTest, meanErrorTest, stdErrorTrain, meanErrorTrain = [], [], [], []

kf = KFold(n_splits=10) # Justify usage of 5 or 10-fold

plt.figure(num=None, figsize=(10, 9), dpi=120)

for Ci in C_range:
    model = Ridge(alpha=1/(2*Ci))
    tempTest, tempTrain = [], []

    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], y[train])
        ypred = model.predict(Xpoly[test])
        ypred_train = model.predict(Xpoly[train])
        
        tempTest.append(mean_squared_error(y[test], ypred))
        tempTrain.append(mean_squared_error(y[train], ypred_train))

    meanErrorTest.append(np.array(tempTest).mean())
    stdErrorTest.append(np.array(tempTest).std())
    meanErrorTrain.append(np.array(tempTrain).mean())
    stdErrorTrain.append(np.array(tempTrain).std())

plt.errorbar(C_range, meanErrorTest, yerr=stdErrorTest, linewidth=3, c="blue", label="Test data")
plt.errorbar(C_range, meanErrorTrain, yerr=stdErrorTrain, linewidth=3, c="orange", label="Training data")
plt.title("Error mean and standard deviation vs C values (Ridge model)")
plt.gca().set(xlabel='C', ylabel="Mean square error")
plt.legend()
plt.show()