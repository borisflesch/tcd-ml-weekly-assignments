#!/usr/bin/env python
# coding: utf-8

# CS7CS4/CSU44061 Machine Learning
# Week 1 Assignment
# Boris Flesch (20300025)
# 
# Downloaded dataset
# id:2-1018.8--10 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (a)(i) Read data
def readData(file):
    """
    Read data from a file and outputs reshaped arrays

    :param file: path to the source file
    :return: 2 numpy arrays X and y
    """
    df = pd.read_csv(file, comment="#")
    X = np.array(df.iloc[:, 0]).reshape(-1,1)
    y = np.array(df.iloc[:, 1]).reshape(-1,1)
    
    return X, y

X, y = readData("week1.csv")

# Visualise data
plt.plot(X, y, '+')
plt.title("Original dataset visualisation")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
#plt.savefig("original-dataset.png")


# (a)(ii) Normalise data
def normaliseData(X):
    """
    Normalise data using average as shift and max-min as scaling factor
    
    :param X: Data array to normalise
    :return: Normalised data, shift value, scaling factor value
    """
    shift = np.average(X)
    scalingFactor = np.max(X) - np.min(X)
    X = (X - shift) / scalingFactor
    return X, shift, scalingFactor

# Linear Regression
def linearRegression(X_in, y_in, alpha=[0.1], theta=[0,0], costThreshold=10e-10, maxIterations=1000):
    """
    Process a linear regression on data
    
    :param X_in: X data (array)
    :param y_in: y data (array)
    :param alpha: Value of the learning rate
    :param theta: Initial values for theta (array)
    :param costThreshold: Cost function limit between two iterations of the gradient descent
    :param maxIterations: Maximum number of iterations for the gradient descent
    :return: theta (mapped to original data, i.e. "de-normalised"), errors (with respect to normalised data), finalError (mapped to original data)
    """
    
    # Copy of input arrays
    X = np.copy(X_in)
    y = np.copy(y_in)
    
    # Normalise data
    X, shiftX, scalingFactorX = normaliseData(X)
    y, shiftY, scalingFactorY = normaliseData(y)
    
    # (a)(iii) Gradient Descent
    costs = []
    m = X.size
    
    for k in range(maxIterations):
        # Theta calculation
        theta[0] += (-2*alpha / m) * sum((theta[0] + theta[1] * X) - y)
        theta[1] += (-2*alpha / m) * sum(((theta[0] + theta[1] * X) - y) * X)
        
        # Cost calculation
        cost = 1/m * sum(np.power((theta[0] + theta[1] * X) - y, 2))
        costs.append(cost)
        
        if (k > 2 and abs(costs[-1] - costs[-2]) <= costThreshold):
            break
            
    costs = np.array(costs)

    # Theta calculation (mapping value to original data)
    theta = [
        theta[0] * scalingFactorY + shiftY - theta[1] * shiftX * scalingFactorY / scalingFactorX,
        theta[1] / scalingFactorX * scalingFactorY
    ] 
    
    # Final cost value (mapping value to original data)
    finalCost = 1/m * sum(np.power((theta[0] + theta[1] * X_in) - y_in, 2))
    
    return theta, costs, finalCost


# (b)(i) Linear regression using different learning rates
alphas = [1, 0.01, 0.001, 0.1]
for alpha in alphas:
    theta, costs, finalCost = linearRegression(X, y, alpha, theta=[0,0], costThreshold=10e-12, maxIterations=1000)
    plt.plot(range(costs.size), costs, '-', label="⍺=" + str(alpha))

plt.legend()
plt.title("Gradient descent's cost function evolution (10e-12 threshold)")
plt.xlabel("Gradient descent iterations")
plt.ylabel("J(θ) (with respect to normalised data)")
plt.show()
#plt.savefig("cost-function-evolution.png")

# (b)(ii)
print("Linear regression model using vanilla Python:")
print("θ₀ = %f ; θ₁ = %f" % (theta[0], theta[1]))

# (b)(iii)
print("J(θ) = %f" % finalCost)

# Baseline model (θ₀ = y mean, θ₁ = 0)
bTheta = [np.mean(y), 0]
bCost = 1/X.size * sum(np.power((bTheta[0] + bTheta[1] * X) - y, 2))
print("J(θ_baseline) = %f\n" % bCost)

# Plot predictions, training data and baseline model
plt.title("Linear regression")
plt.xlabel("Input X")
plt.ylabel("Output y")
plt.scatter(X, y, marker='+', label="Training data")
plt.plot(X, theta[0] + theta[1] * X, 'r-', linewidth=2, label="Predictions")
plt.plot(X, bTheta[0] + bTheta[1] * X, color='g', linestyle='dashed', linewidth=2, label="Baseline model (y mean)")
plt.legend()
plt.show()
#plt.savefig("linear-regression.png")


# (b)(iv) Linear Regression model with sklearn
from sklearn.linear_model import LinearRegression

X, y = readData("week1.csv")

model = LinearRegression().fit(X, y)
cost = 1/X.size * sum(np.power((model.intercept_ + model.coef_ * X) - y, 2))

print("Linear regression model using sklearn:")
print("θ₀ = %f ; θ₁ = %f" % (model.intercept_, model.coef_))
print("J(θ) = %f" % cost)

plt.title("Linear regression using sklearn")
plt.scatter(X, y, marker='+', label="Training data")
plt.plot(X, model.intercept_ + model.coef_ * X, 'r-', linewidth=2, label="Predictions")
plt.legend()
plt.xlabel("Input X")
plt.ylabel("Output y")
plt.show()
#plt.savefig("linear-regression-sklearn.png")