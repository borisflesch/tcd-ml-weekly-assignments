from preprocessing.data_preprocessing import parse_data
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import multilabel_confusion_matrix
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

# PARAMETERS

performCrossValidationN = False
performCrossValidationC = False
model_n = (25,)
model_C = 10

# READ DATA

X, y = parse_data(type='matrix')
X = X[1]
y = y[1]

# SPLIT DATA

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

print("Train size:", len(Xtrain))
print("Test size:", len(Xtest))


def printVerifTable(yTrue, yPred):
    prec = precision_recall_fscore_support(yTrue, yPred)
    precAvg = precision_recall_fscore_support(yTrue, yPred, average="micro")
    acc = accuracy_score(yTrue, yPred)
    print(tabulate([
        ['0', prec[0][0], prec[0][1], prec[0][2]],
        ['1', prec[1][0], prec[1][1], prec[1][2]],
        ['2', prec[2][0], prec[2][1], prec[2][2]],
        ['avg/total', precAvg[0], precAvg[1], precAvg[2], acc]
    ], headers=['', 'precision', 'recall', 'f1-score', 'accuracy'], tablefmt='orgtbl'))


if performCrossValidationN:
    print("\n=== CROSS VALIDATION FOR n ===\n")
    precRecallF1Class1 = []; precRecallF1Class2 = []; precRecallF1Class3 = []
    stdErr1 = []; stdErr2 = []; stdErr3 = []
    hidden_layer_range = [5, 10, 25, 50, 75, 100]
    #hidden_layer_range = [2, 3, 5]
    for n in hidden_layer_range:
        print("> hidden layer size %d" % n)
        kf = KFold(n_splits=5)
        temp1 = []; temp2 = []; temp3 = []
        meanAccuracy = []
        for train, test in kf.split(Xtrain):
            model = MLPClassifier(hidden_layer_sizes=n, alpha=1.0/(2*model_C), max_iter=100000)
            model.fit(Xtrain[train], ytrain[train])
            ypred = model.predict(Xtrain[test])
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

    plt.errorbar(hidden_layer_range, precRecallF1Class1[:, 2], yerr=stdErr1[:, 2], label='Class 1')
    plt.errorbar(hidden_layer_range, precRecallF1Class2[:, 2], yerr=stdErr2[:, 2], label='Class 2')
    plt.errorbar(hidden_layer_range, precRecallF1Class3[:, 2], yerr=stdErr3[:, 2], label='Class 3')
    plt.title("n cross-validation")
    plt.xlabel('n (#hidden layer nodes)')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.show()


if performCrossValidationC:
    print("\n=== CROSS VALIDATION FOR C ===\n")
    precRecallF1Class1 = []; precRecallF1Class2 = []; precRecallF1Class3 = []
    stdErr1 = []; stdErr2 = []; stdErr3 = []
    # C_range = [0.1, 1, 5, 10, 50, 100, 500]
    C_range = [1, 5, 10, 50]
    for C in C_range:
        print("> C %.1f" % C)
        kf = KFold(n_splits=5)
        temp1 = []; temp2 = []; temp3 = []
        meanAccuracy = []
        for train, test in kf.split(Xtrain):
            model = MLPClassifier(hidden_layer_sizes=model_n, alpha=1.0/(2*C), max_iter=100000)
            model.fit(Xtrain[train], ytrain[train])
            ypred = model.predict(Xtrain[test])
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

    plt.errorbar(C_range, precRecallF1Class1[:, 2], yerr=stdErr1[:, 2], label='Class 1')
    plt.errorbar(C_range, precRecallF1Class2[:, 2], yerr=stdErr2[:, 2], label='Class 2')
    plt.errorbar(C_range, precRecallF1Class3[:, 2], yerr=stdErr3[:, 2], label='Class 3')
    plt.title("C cross-validation")
    plt.xlabel('C')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.show()


print("\n=== FINAL RESULTS ===\n")

model = MLPClassifier(hidden_layer_sizes=model_n, alpha=1.0/(2*model_C), max_iter=100000).fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
print("Model classes:", model.classes_)
print("Model confusion matrices:")
print(multilabel_confusion_matrix(ytest, ypred))
printVerifTable(ytest, ypred)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
model = MLPClassifier(hidden_layer_sizes=model_n, alpha=1.0/(2*model_C), max_iter=100000).fit(Xtrain, ytrain)
pred_prob = model.predict_proba(Xtest)
for i in range(len(model.classes_)):
    fpr[i], tpr[i], _ = roc_curve(ytest[:, i], pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(model.classes_)):
    plt.plot(fpr[i], tpr[i], label='Class %d ROC curve (area = %0.2f)' % ((i+1), roc_auc[i]))
    print("AUC for Class", (i+1), "=", roc_auc[i])

pl1=plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (C=' + str(model_C) + ', n=' + str(model_n) + ') for each class')
plt.legend(loc="lower right")
plt.show()



# Compute ROC curve and ROC area for all classes
classifier = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=model_n, alpha=1.0/(2*model_C), max_iter=100000))
yproba = classifier.fit(Xtrain, ytrain).predict_proba(Xtest)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classifier.classes_)):
    fpr[i], tpr[i], _ = roc_curve(ytest[:, i], yproba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ytest.ravel(), yproba.ravel())
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
plt.title('ROC Curve (C=' + str(model_C) + ', n=' + str(model_n) + ')')
plt.legend(loc="lower right")
plt.show()


print("\n=== COMPARISON WITH BASELINE ===\n")

dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
ydummy = dummy.predict(Xtest)
print("Model classes:", model.classes_)
print("Baseline confusion matrices:")
print(multilabel_confusion_matrix(ytest, ydummy))