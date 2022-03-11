import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,precision_recall_fscore_support,multilabel_confusion_matrix,roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold ,train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from preprocessing.data_preprocessing import parse_data

x,y = parse_data(type='number')
x= x[1]
y= y[1]
Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


print(ytrain)

findQ = False 
if(findQ):  
    acc=[]
    std_error=[]
    polys = list(range(1,4))
    precRecallF1Class1 = []; precRecallF1Class2 = []; precRecallF1Class3 = []
    stdErr1 = []; stdErr2 = []; stdErr3 = []
    
    for q in polys:
        
        Xpoly = PolynomialFeatures(q).fit_transform(Xtrain)  
        print("> Poly feature order %.1f" % q)
        kf = KFold(n_splits=5)
        temp1 = []; temp2 = []; temp3 = []
        meanAccuracy = []
        for train, test in kf.split(Xpoly):
            model = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=100000, C=0.1).fit(Xpoly[train],y[train])
            model.fit(Xpoly[train], ytrain[train])
            ypred = model.predict(Xpoly[test])
            prec = np.array(precision_recall_fscore_support(ytrain[test], ypred))
            temp1.append(prec[0]); temp2.append(prec[1]); temp3.append(prec[2])
            meanAccuracy.append(accuracy_score(ytrain[test], ypred))

        print("\tAccuracy = %f\n" % np.array(meanAccuracy).mean())
        precRecallF1Class1.append(np.array(temp1).mean(axis=0))
        precRecallF1Class2.append(np.array(temp2).mean(axis=0))
        precRecallF1Class3.append(np.array(temp3).mean(axis=0))
        stdErr1.append(np.array(temp1).std(axis=0))
        stdErr2.append(np.array(temp2).std(axis=0))
        stdErr3.append(np.array(temp3).std(axis=0))

    precRecallF1Class1 = np.array(precRecallF1Class1)
    precRecallF1Class2 = np.array(precRecallF1Class2)
    precRecallF1Class3 = np.array(precRecallF1Class3)
    stdErr1 = np.array(stdErr1)
    stdErr2 = np.array(stdErr2)
    stdErr3 = np.array(stdErr3)

    plt.errorbar(polys, precRecallF1Class1[:, 2], yerr=stdErr1[:, 2], label='Class 1')
    plt.errorbar(polys, precRecallF1Class2[:, 2], yerr=stdErr2[:, 2], label='Class 2')
    plt.errorbar(polys, precRecallF1Class3[:, 2], yerr=stdErr3[:, 2], label='Class 3')
    plt.title("Polynomial features cross-validation")
    plt.xlabel('Polynomial features')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.show()


findC = False
if(findC):

    precRecallF1Class1 = []; precRecallF1Class2 = []; precRecallF1Class3 = []
    stdErr1 = []; stdErr2 = []; stdErr3 = []
    cis = [0.001,0.01,0.1,1]
    
    Xpoly = PolynomialFeatures(1).fit_transform(Xtrain)  
    for ci in cis:
        print("> C value  %.1f" % ci)
        kf = KFold(n_splits=5)
        temp1 = []; temp2 = []; temp3 = []
        meanAccuracy = []
        for train, test in kf.split(Xpoly):
            model = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=100000, C=ci).fit(Xpoly[train],y[train])
            model.fit(Xpoly[train], ytrain[train])
            ypred = model.predict(Xpoly[test])
            prec = np.array(precision_recall_fscore_support(ytrain[test], ypred))
            temp1.append(prec[0]); temp2.append(prec[1]); temp3.append(prec[2])
            meanAccuracy.append(accuracy_score(ytrain[test], ypred))

        print("\tAccuracy = %f\n" % np.array(meanAccuracy).mean())
        precRecallF1Class1.append(np.array(temp1).mean(axis=0))
        precRecallF1Class2.append(np.array(temp2).mean(axis=0))
        precRecallF1Class3.append(np.array(temp3).mean(axis=0))
        stdErr1.append(np.array(temp1).std(axis=0))
        stdErr2.append(np.array(temp2).std(axis=0))
        stdErr3.append(np.array(temp3).std(axis=0))

    precRecallF1Class1 = np.array(precRecallF1Class1)
    precRecallF1Class2 = np.array(precRecallF1Class2)
    precRecallF1Class3 = np.array(precRecallF1Class3)
    stdErr1 = np.array(stdErr1)
    stdErr2 = np.array(stdErr2)
    stdErr3 = np.array(stdErr3)

    plt.errorbar(cis, precRecallF1Class1[:, 2], yerr=stdErr1[:, 2], label='Class 1')
    plt.errorbar(cis, precRecallF1Class2[:, 2], yerr=stdErr2[:, 2], label='Class 2')
    plt.errorbar(cis, precRecallF1Class3[:, 2], yerr=stdErr3[:, 2], label='Class 3')
    plt.title("C cross-validation")
    plt.xlabel('C')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.show()


good_poly= 1
good_c=0.1


# Compute ROC curve and ROC area for all classes
classifier = OneVsRestClassifier(LogisticRegression(penalty='l2',solver='lbfgs',max_iter=100000, C=good_c))
yproba = classifier.fit(Xtrain, ytrain).predict_proba(Xtest)

ytestROC = label_binarize(ytest, classes=[0,1,2])
n_classes = 3

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _  = roc_curve(ytestROC[:, i], yproba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ytestROC[:, i], yproba[:, i])
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (C=' + str(good_c) + ', Poly=' + str(good_poly) + ')')
plt.legend(loc="lower right")
plt.show()

model = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=100000, C=good_c).fit(Xtrain,ytrain)
ypred = model.predict(Xtest)

print("LR confusion matrix:")
print(multilabel_confusion_matrix(ytest, ypred))
print( accuracy_score(ytest, ypred))



print("\n=== COMPARISON WITH BASELINE ===\n")

dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
ydummy = dummy.predict(Xtest)
print("Model classes:", model.classes_)
print("Baseline confusion matrix:")
print(multilabel_confusion_matrix(ytest, ydummy))
print( accuracy_score(ytest, ydummy))





