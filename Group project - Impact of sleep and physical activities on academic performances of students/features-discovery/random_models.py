
import numpy as np
import math
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import random
from scipy.interpolate import griddata

def create_permutation(df,num_features):
    x = []
    x_indexes = []
    for j in range(num_features):
        index = random.randint(0,len(df.columns)-1)
        
        good_value=False
        
        while not good_value:
            good_value=True
            for z in x_indexes:
                if(abs(z-index)<5):
                    index = random.randint(0,len(df.columns)-1)
                    good_value=False
            
        x_indexes.append(index)
        x.append(df.iloc[:,index])        
    y_index = random.randint(0,len(df.columns)-1)


    good_value=False
    while not good_value:
        good_value=True
        for z in x_indexes:
            if(abs(z-y_index)<5):
                y_index = random.randint(0,len(df.columns)-1)
                good_value=False
        


    y = df.iloc[:,y_index]
    x = np.array(x).transpose()
    y = np.array(y).reshape(-1,1)
    return x,y,(x_indexes,y_index)

def create_3Dgraph(current_indexes,poly_order,p_model,p_dummy,inputs,output):
    
    Xtest=[]
    grid = np.linspace(-1.5,1.5)
    for i in grid:
        for j in grid:
            Xtest.append([i,j])
    Xtest= np.array(Xtest)

    Xtestpoly = PolynomialFeatures(poly_order).fit_transform(Xtest)

    predValuesGrid = p_model.predict(Xtestpoly)
    predValuesDummy = p_dummy.predict(Xtestpoly)


    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(inputs[:,0],inputs[:,1],output,s=30)

    X,Y= np.meshgrid(grid, grid)

    Z = griddata((Xtest[:,0], Xtest[:,1]), predValuesGrid, (X.flatten(), Y.flatten()), 'nearest').reshape(50,50)
    Zdummy = griddata((Xtest[:,0], Xtest[:,1]), predValuesDummy, (X.flatten(), Y.flatten()), 'nearest').reshape(50,50)

    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, color="red")
    ax.plot_wireframe(X, Y, Zdummy, rstride=5, cstride=5, color="green")

    ax.legend(["Training Data","Predictions","Baseline"],  loc='upper left',)
    ax.set_xlabel("Feature 1(x1)"); ax.set_ylabel("Feature 2(x2)"); ax.set_zlabel("output(y)")
    #plt.title("Lasso Regression C="+str(C))
    plt.show()

  
  

df=pd.read_csv("../dataset/_old_polished.csv", header=None)
# Academic data is df[27]
col_names = df.iloc[0,:]
print(col_names)
df = df.drop([0])

num_features = 2
max_poly_features = 3
interesting_increse=100

total_iter=100000

for i in range(total_iter):
    itera = i
    #if((iter % (total_iter/100))==0):
    if(itera% math.floor(total_iter/100)==0):
        print("Done:" + str(itera/math.floor(total_iter/100)) + "%")

    features,output,current_indexes=create_permutation(df,num_features)
    
    features = normalize(features,axis=0,norm='l2')
    output = normalize(output,axis=0,norm='l2')
    
    for poly in range(1,max_poly_features+1):
        temp=[]
        Xpoly = PolynomialFeatures(poly).fit_transform(features)  

        #print(features.shape)
        #print(output.shape)
        dummy_mean = DummyRegressor(strategy='mean').fit(Xpoly,output)
        preds = dummy_mean.predict(Xpoly)
        acc_mean = mean_squared_error(output,preds)
        #print(acc_mean)
        dummy_median = DummyRegressor(strategy='median').fit(Xpoly,output)
        preds_med = dummy_median.predict(Xpoly)
        acc_med = mean_squared_error(output,preds_med)
        #print(acc_med)

        kf = KFold(n_splits=10)
        for train,test in kf.split(Xpoly):
            model = LinearRegression().fit(Xpoly[train],output[train])
            ypred = model.predict(Xpoly[test])
            temp.append(mean_squared_error(output[test],ypred))
            #print("Done")
        acc= (np.array(temp).mean())
        std_error = (np.array(temp).std())
        if(acc<acc_mean and acc< acc_med):
            if(acc_mean<acc_med):
                small_acc=acc_mean
                small_model= dummy_mean
            else:
                small_acc=acc_med
                small_model= dummy_median
            increse= ((small_acc-acc)/acc)*100

            if(increse>interesting_increse):
                #print("Found Interesting!")
                print("["+col_names[current_indexes[0][0]]+","+col_names[current_indexes[0][1]]+"],"+col_names[current_indexes[1]])
                print(current_indexes)
                print("Increase is " + str(increse) + ", from "+ str(small_acc)+ " to "+str(acc)+ " with poly "+ str(poly))
                #create_3Dgraph(current_indexes,poly,model,small_model,features,output)


           

