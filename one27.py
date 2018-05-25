import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#Loading the dataset
url="wine_dataset.csv"
names=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality','style']

#Converting Object to float for mathematical caluclations
dataset=pd.read_csv(url,converters={"quality":float})

#Size of the Dataset
print(dataset.shape)

#Assuring the types of data
print(dataset.dtypes)

#Analysing the dependency between the variables using Pearson Coefficient
data=dataset.corr(method='pearson')
pd.set_option('precision', 2)
data.to_csv("pearson.csv", index=True, encoding='utf8')
print("File created")

#Analysing the dependency between the variables using Spearman Coefficient
data1=dataset.corr(method='spearman')
pd.set_option('precision', 2)
data1.to_csv("spearman.csv", index=True, encoding='utf8')
print("File created")

#Dividing the dataset for training,testing and validation as per Pareto Principle
array=dataset.values
X=array[:,0:12]
Y=array[:,12]
validation_size=0.20
seed=7
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

# Make predictions on validation dataset using KNN
print('KNN\n')
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions=knn.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

# Make predictions on validation dataset using LR
print('LR\n')
knn1=LogisticRegression()
knn1.fit(X_train,Y_train)
predictions=knn1.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

# Make predictions on validation dataset using LDA
print('LDA\n')
knn5=LinearDiscriminantAnalysis()
knn5.fit(X_train,Y_train)
predictions=knn5.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))


# Make predictions on validation dataset using CART
print('CART\n')
knn2=DecisionTreeClassifier()
knn2.fit(X_train,Y_train)
predictions=knn2.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

# Make predictions on validation dataset using Gaussian Naive Bayesian
print('GaussianNB\n')
knn3=GaussianNB()
knn3.fit(X_train,Y_train)
predictions=knn3.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

# Make predictions on validation dataset using SVM
print('SVM\n')
knn4=SVC()
knn4.fit(X_train,Y_train)
predictions=knn4.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

#Seed Input
seed=7
scoring='accuracy'
# Spot Check Algorithms
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
# evaluate each model in turn
results=[]
names=[]
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)

# Compare Algorithms
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
