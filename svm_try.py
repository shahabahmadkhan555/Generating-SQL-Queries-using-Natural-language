from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import learning_curve
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt 
fname = 'train'
X = []
y = []

with open(fname) as f:
    for line in f:
        record = line.split(',')
        X.append(record[0] + ' ' + record[1])
        y.append(int(record[-1]))

transformer = CountVectorizer(stop_words="english", binary=True)
x= transformer.fit_transform(X)  #training data 
y= np.array(y)

a=x.toarray()
rows = a.shape[0]
cols = a.shape[1]
ct=0
#clf=SVC(kernel='rbf', probability=True,gamma=float(1.0/10**6))
clf=SVC(kernel='poly',probability=True)

score = cross_val_score(clf, x, y, cv=3, scoring='roc_auc')

#train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, cv=5,scoring='roc_auc')
#train_scores, valid_scores = validation_curve(Ridge(), x, y,np.logspace(-7, 3, 3),cv=5)


print ("Score",np.round(np.mean(score), 2))
print ("STD", np.round(np.std(score), 3))

print()
print()
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
#clf.fit(x_train, y_train)
#print(type(x_test))
#y_pred = clf.predict(x_test)
#print ('accuracy',accuracy_score(y_test, y_pred, normalize=True))

param_range = np.logspace(-6, -1, 10)
train_scores, test_scores = validation_curve(clf, x, y, param_name="gamma", param_range=param_range, cv=3, scoring="roc_auc", n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
print (test_scores_mean)
plt.title("Validation Curve with SVM and kernel as rbf")
plt.xlabel("gamma")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",color="orangered",lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std,color="lightsalmon",lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",color="navy",lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std,color="lightskyblue",lw=lw)
plt.legend(loc="best")
plt.show()