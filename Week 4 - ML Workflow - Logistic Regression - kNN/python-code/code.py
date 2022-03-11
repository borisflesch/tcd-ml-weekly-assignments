#!/usr/bin/env python
# coding: utf-8

# CS7CS4/CSU44061 Machine Learning
# Week 4 Assignment
# Boris Flesch (20300025)
# 
# Downloaded dataset
# id:23-46--23-0
# id:23--23-23-0


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# (a)
def readData(filepath):
    df = pd.read_csv(filepath, comment="#")
    X1 = df.iloc[:,0]
    X2 = df.iloc[:,1]
    X = np.column_stack((X1,X2))
    y = df.iloc[:,2]
    return X,y

def plotData(X,y):
    X_m1 = X[np.where(y == -1)]
    X_p1 = X[np.where(y == 1)]
    plt.scatter(X_m1[:, 0], X_m1[:, 1], c='r', marker='+', label="y = -1")
    plt.scatter(X_p1[:, 0], X_p1[:, 1], c='b', marker='+', label="y = 1")
    plt.gca().set(title="Training data visualisation", xlabel="X1", ylabel="X2")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()

# (a)(i)
def plotRangeQandC(X, y, q_range, C_range):
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    
    for Ci in C_range:
        model = LogisticRegression(penalty="l2", C=Ci, max_iter=1000)
        mean_error, std_error = [], []
        for qi in q_range:
            Xpoly = PolynomialFeatures(qi).fit_transform(X)
            scores = cross_val_score(model, Xpoly, y, cv=5, scoring='f1')
            mean_error.append(np.array(scores).mean())
            std_error.append(np.array(scores).std())

        plt.errorbar(q_range, mean_error, yerr=std_error, linewidth=3, label="C = %.3f"%Ci)
        
    plt.title("F1 Score and standard deviation vs q values (Logistic Regression)")
    plt.gca().set(xlabel='q', ylabel="F1 Score")
    plt.legend()
    plt.show()

# (a)(ii)
def plotRangeC(X, y, q, C_range, printParameters=False):
    Xpoly = PolynomialFeatures(q).fit_transform(X)
    mean_error, std_error = [], []
    for Ci in C_range:
        model = LogisticRegression(penalty="l2", C=Ci, max_iter=1000)
        scores = cross_val_score(model, Xpoly, y, cv=5, scoring='f1')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        
        if (printParameters):
            model.fit(Xpoly, y)
            theta = np.insert(model.coef_, 0, model.intercept_)
            print("C = %.1f"%Ci)
            print("θ =", theta)

    fig = plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.errorbar(C_range, mean_error, yerr=std_error, linewidth=3)
    plt.title("F1 Score and standard deviation vs C values (Logistic Regression)")
    plt.gca().set(xlabel='C', ylabel="F1 Score")
    plt.show()

def plotPredictions(X, y, q, model, title="Prediction surface on test data"):
    Xpoly = PolynomialFeatures(q).fit_transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xpoly,y,test_size=0.2)
    Xtest_m1 = Xtest[np.where(ytest == -1)]
    Xtest_p1 = Xtest[np.where(ytest == 1)]

    model.fit(Xtrain, ytrain)
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    meshStep = .01
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = Xtest[:, 1].min() - 1, Xtest[:, 1].max() + 1
    y_min, y_max = Xtest[:, 2].min() - 1, Xtest[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, meshStep), np.arange(y_min, y_max, meshStep))
    Xtest = np.c_[xx.ravel(), yy.ravel()]
    Xtest = PolynomialFeatures(q).fit_transform(Xtest)
    Z = model.predict(Xtest).reshape(xx.shape)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(Xtest_m1[:, 1], Xtest_m1[:, 2], c='r', marker='+', label="y = -1")
    plt.scatter(Xtest_p1[:, 1], Xtest_p1[:, 2], c='b', marker='+', label="y = 1")

    plt.gca().set(title=title, xlabel='X1', ylabel="X2")

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(mpatches.Patch(color='#FFAAAA', label='y_pred = -1'))
    handles.append(mpatches.Patch(color='#AAAAFF', label='y_pred = 1'))
    plt.legend(handles=handles)
    plt.show()


# (b)
def plotRangeKandQ(X, y, q_range, k_range):
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80)
    q_range = [1,2,3]
    for qi in q_range:
        Xpoly = PolynomialFeatures(qi).fit_transform(X)

        # k for kNN cross-validation
        mean_f1, std_f1 = [], []
        for ki in k_range:
            model = KNeighborsClassifier(n_neighbors=ki,weights='uniform')
            scores = cross_val_score(model, Xpoly, y, cv=5, scoring='f1') # cv -> KFold
            mean_f1.append(np.array(scores).mean())
            std_f1.append(np.array(scores).std())

        plt.errorbar(k_range, mean_f1, yerr=std_f1, linewidth=3, label='q = %d'%qi)

    plt.title("F1 Score and standard deviation vs k values (kNN)")
    plt.gca().set(xlabel='k', ylabel="F1 Score")
    plt.legend()
    plt.show()


# (c)
def confusionMatrices(X, y, q, models):
    Xpoly = PolynomialFeatures(q).fit_transform(X)
    # Use the same split for each model tested
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xpoly,y,test_size=0.2)
    # cross_val_predict
    
    for model, title in models:
        model.fit(Xtrain, ytrain)
        disp = plot_confusion_matrix(model, Xtest, ytest, display_labels=["y = -1", "y = 1"], cmap=plt.cm.Blues, values_format = 'd')
        disp.ax_.set_title("Confusion matrix for " + title)
        plt.show()


# (d)
def plotRocCurves(X, y, q, C, k):
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80)
    
    Xpoly = PolynomialFeatures(q).fit_transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xpoly,y,test_size=0.2)

    model = LogisticRegression(penalty="l2", C=100).fit(Xtrain, ytrain)
    # decision_function: Predict confidence scores for samples.
    fpr, tpr, _ = roc_curve(ytest,model.decision_function(Xtest))
    plt.plot(fpr,tpr, label='Logistic Regression (q = %d, C = %.3f)'%(q,C))

    model = KNeighborsClassifier(n_neighbors=k,weights='uniform').fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest,model.predict_proba(Xtest)[:,1]) 
    plt.plot(fpr,tpr, label='kNN (k = %d)'%k)

    model = DummyClassifier(strategy='most_frequent').fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest,model.predict_proba(Xtest)[:,1]) 
    plt.plot(fpr,tpr, label='Baseline classifier (most frequent)')
    
    model = DummyClassifier(strategy="uniform").fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest,model.predict_proba(Xtest)[:,1]) 
    plt.plot(fpr,tpr, label='Baseline classifier (random)')

    plt.title("ROC Curves of classifiers")
    plt.gca().set(xlabel='False positive rate', ylabel="True positive rate")
    plt.legend()
    plt.show()



#######
# (i) #
#######

X,y = readData("week4-1.csv")
plotData(X,y)

# (a)(i) Choose value of q
plotRangeQandC(X, y, q_range=[1,2,3], C_range=[0.001, 1, 1000])

# Check with predictions:
model = LogisticRegression(penalty="l2", C=1)
plotPredictions(X, y, q=1, model=model, title="Prediction surface on test data (LogisticRegression, q=1, C=1)")
plotPredictions(X, y, q=2, model=model, title="Prediction surface on test data (LogisticRegression, q=2, C=1)")
# plotPredictions(X, y, q=10, model=model, title="Prediction surface on test data (LogisticRegression, q=3, C=1)")

# (a)(ii)
plotRangeC(X, y, q=2, C_range=[1, 5, 10, 50, 100, 500, 1000])

model = LogisticRegression(penalty="l2", C=1)
plotPredictions(X, y, q=2, model=model, title="Prediction surface on test data (LogisticRegression, q=2, C=1)")
model = LogisticRegression(penalty="l2", C=100)
plotPredictions(X, y, q=2, model=model, title="Prediction surface on test data (LogisticRegression, q=2, C=100)")


# (b)
plotRangeKandQ(X, y, q_range=[1,2,3], k_range=[2,3,5,7,10,13,15,17,20])

model = KNeighborsClassifier(n_neighbors=5,weights='uniform')
plotPredictions(X, y, q=2, model=model, title="Prediction surface on test data (kNN, k=5)")


# (c)
# print(np.unique(y, return_counts=True))
models = [
            [LogisticRegression(penalty="l2", C=100), 'Logistic Regression (q = 2, C = 100)'],
            [KNeighborsClassifier(n_neighbors=5,weights='uniform'), 'kNN (k = 5)'],
            [DummyClassifier(strategy="most_frequent"), 'Baseline Classifier (most frequent)'],
            [DummyClassifier(strategy="uniform"), 'Baseline Classifier (random)']
         ]

confusionMatrices(X, y, 2, models)

# (d)
plotRocCurves(X, y, q=2, C=100, k=5)



########
# (ii) #
########
X,y = readData("week4-2.csv")
plotData(X,y)

# (a)(i)
plotRangeQandC(X, y, q_range=[1,2,3,4,5], C_range=[0.001, 1, 1000])

model = LogisticRegression(penalty="l2", C=1)
plotPredictions(X, y, q=1, model=model, title="Prediction surface on test data (LogisticRegression, q=1, C=1)")
plotPredictions(X, y, q=2, model=model, title="Prediction surface on test data (LogisticRegression, q=2, C=1)")

# (a)(ii)
plotRangeC(X, y, q=1, C_range=[0.1, 1, 10, 100], printParameters=True)

model = LogisticRegression(penalty="l2", C=1)
plotPredictions(X, y, q=1, model=model, title="Prediction surface on test data (LogisticRegression, q=1, C=1)")
model = LogisticRegression(penalty="l2", C=100)
plotPredictions(X, y, q=1, model=model, title="Prediction surface on test data (LogisticRegression, q=1, C=100)")


# (b)
plotRangeKandQ(X, y, q_range=[1,2,3], k_range=[1,2,3,4,5,6,7,8,9,10])
plotRangeKandQ(X, y, q_range=[1,2,3], k_range=[1,10,100,500]) # max nsamples = 754

model = KNeighborsClassifier(n_neighbors=100,weights='uniform')
plotPredictions(X, y, q=1, model=model, title="Prediction surface on test data (kNN, k=100)")


# (c)
models = [
            [LogisticRegression(penalty="l2", C=1), 'Logistic Regression (q = 1, C = 1)'],
            [KNeighborsClassifier(n_neighbors=100,weights='uniform'), 'kNN (k = 100)'],
            [DummyClassifier(strategy="most_frequent"), 'Baseline Classifier (most frequent)'],
            [DummyClassifier(strategy="uniform"), 'Baseline Classifier (random)']
         ]
confusionMatrices(X, y, q=1, models=models)


# (d)
plotRocCurves(X, y, q=1, C=1, k=100)


# Purely for testing purposes (seeing the impact of a very small value for k)
# model = KNeighborsClassifier(n_neighbors=1,weights='uniform')
# plotPredictions(X, y, q=1, model=model, title="Prediction surface on test data (kNN, k=100)")
# plotRocCurves(X, y, q=1, C=1, k=3)