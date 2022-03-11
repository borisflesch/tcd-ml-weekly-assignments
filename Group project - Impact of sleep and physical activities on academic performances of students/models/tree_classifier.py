import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from preprocessing.data_preprocessing import parse_data
from sklearn.tree import DecisionTreeClassifier

x, y = parse_data(type='number')
Xtrain, Xtest, ytrain, ytest = train_test_split(x[1],y[1],test_size=0.2)

# Import the library for Tree Classifier


# CrossValidation on Tree Depth
precRecallF1Class1 = []; precRecallF1Class2 = []; precRecallF1Class3 = []
stdErr1 = []; stdErr2 = []; stdErr3 = []
tree_depth = [1,10,100,1000]
for t in tree_depth:
    print("> Tree Depth %.1f" % t)
    kf = KFold(n_splits=5)
    temp1 = []; temp2 = []; temp3 = []
    meanAccuracy = []
    for train, test in kf.split(Xtrain):
        model = TreeClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=t, random_state=42)
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

plt.errorbar(tree_depth, precRecallF1Class1[:, 2], yerr=stdErr1[:, 2], label='Class 1')
plt.errorbar(tree_depth, precRecallF1Class2[:, 2], yerr=stdErr2[:, 2], label='Class 2')
plt.errorbar(tree_depth, precRecallF1Class3[:, 2], yerr=stdErr3[:, 2], label='Class 3')
plt.title("Tree Depth cross-validation")
plt.xlabel('Tree depth')
plt.ylabel('F1-Score')
plt.legend()
plt.show()




# Tree Classifier with cross validated tree depth
TreeClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=100, random_state=42)
# Fit the model on the training data
TreeClassifier.fit(Xtrain, ytrain.reshape(-1,1))


# Obtain the predictions on training data
ypred_classifier=TreeClassifier.predict(Xtest)

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


# Accuracy results on test set

accuracy = metrics.accuracy_score(ytest, ypred_classifier)
print("Accuracy of Decision Tree: {:.2f}".format(accuracy))

# Print the confusion matrix
print("Model confusion matrix:")
cm=multilabel_confusion_matrix(ytest, ypred_classifier)

print('Confusion Matrix of Tree Classifier: \n', cm)

# Baseline
from sklearn.dummy import DummyClassifier
Dummy=DummyClassifier(strategy="most_frequent")
Dummy.fit(Xtrain,ytrain.reshape(-1,1))
ypred_dummy=Dummy.predict(Xtest)
accuracy = metrics.accuracy_score(ytest, ypred_dummy)
print("Accuracy of Dummy Classifier: {:.2f}".format(accuracy))
# Confusion Matrix of baseline
cmdummy=multilabel_confusion_matrix(ytest,ypred_dummy)
print('Confusion Matrix of Baseline: \n', cmdummy)


# ROC curve for each class

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
ytestROC = label_binarize(ytest, classes=[0,1,2])
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()
pred_prob = TreeClassifier.predict_proba(Xtest)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ytestROC[:, i], pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
AUC=[]
    
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    pl1=plt.plot([0, 1], [0, 1], color='green',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class'+str(i+1))
    plt.legend(loc="lower right")
    plt.show()
    AUC.append(auc(fpr[i], tpr[i]))
print("\n=== AUC of Classes 1,2,3 ===\n")
print(AUC)

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ytestROC.ravel(), pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], label='ROC curve (area = %0.2f)' % roc_auc["micro"])
pl1=plt.plot([0, 1], [0, 1], color='green',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
