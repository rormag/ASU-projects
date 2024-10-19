# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:51:26 2024

@author: rorym
"""

from m5_plotdr import plot_decision_regions            # plotting function
import matplotlib.pyplot as plt                        # so we can add to plot
from sklearn import datasets                           # read the data sets
import numpy as np                                     # needed for arrays
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import Perceptron            # the algorithm
from sklearn.metrics import accuracy_score             # grade the results
import pandas as pd
heart_df = pd.read_csv('heart1.csv')                 # load the data set



                      ############
                      #PERCEPTRON#
                      ############


heart_copy = heart_df.copy()
X =  heart_copy.drop('a1p2',axis=1)                      # separate the features we want
y = heart_df['a1p2']                              # extract the classifications

# split the problem into train and test
# this will yield 70% training and 30% test
# random_state allows the split to be reproduced
# stratify=y not used in this case
X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)

# scale X by removing the mean and setting the variance to 1 on all features.
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.
# (mean and standard deviation may be overridden with options...)

sc = StandardScaler()                    # create the standard scalar
sc.fit(X_train)                          # compute the required transformation
X_train_std = sc.transform(X_train)      # apply to the training data
X_test_std = sc.transform(X_test)        # and SAME transformation of test data



ppn = Perceptron(max_iter=200, tol=1e-1, eta0=0.01,
                 fit_intercept=True, random_state=0, verbose=False) 
ppn.fit(X_train_std, y_train)              # do the training

#print('Number in test ',len(y_test))
y_pred = ppn.predict(X_test_std)          # now try with the test data
# Note that this only counts the samples where the predicted value was wrong
print('PERCEPTRON')  # how'd we do?
print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))

 
y_train_pred = ppn.predict(X_train_std) 
print('Training Accuracy: %.3f' % accuracy_score(y_train, y_train_pred))


X_combined = np.vstack((X_train_std, X_test_std)) #combining test and training set to compare
y_combined = np.hstack((y_train, y_test))
y_combined_pred = ppn.predict(X_combined)
print('Combined Accuracy: %.3f' % \
        accuracy_score(y_combined, y_combined_pred))








                        #####################
                        #LOGISTIC REGRESSION#
                        #####################



from sklearn.linear_model import LogisticRegression 


X_train, X_test, y_train, y_test = \
                 train_test_split(X,y,test_size=0.3,random_state=0)


lr = LogisticRegression(C=1, solver='sag', \
                            random_state=0)
lr.fit(X_train_std, y_train)         # apply the algorithm to training data

    # combine the train and test data
lr.score(X_test,y_test)
y_pred=lr.predict(X_test_std)




print('LOGISTIC REGRESSION')
  # how'd we do?
print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))

y_train_pred = lr.predict(X_train_std) 
print('Training Accuracy: %.3f' % accuracy_score(y_train, y_train_pred))


X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = lr.predict(X_combined)
print('Combined Accuracy: %.3f' % \
        accuracy_score(y_combined, y_combined_pred))





                                #####
                                #SVM#
                                #####
                                
from sklearn.svm import SVC                          # the algorithm


X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)



svm = SVC(kernel='linear', C=.1, random_state=0)
svm.fit(X_train_std, y_train)                      # do the training

y_pred = svm.predict(X_test_std)                   # work on the test data

    # show the results
   
print('SUPPORT VECTOR MACHINES')
print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))

y_train_pred = svm.predict(X_train_std) 
print('Training Accuracy: %.3f' % accuracy_score(y_train, y_train_pred))


X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = svm.predict(X_combined)
print('Combined Accuracy: %.3f' % \
        accuracy_score(y_combined, y_combined_pred))


                                ########################
                                #DECISION TREE LEARNING#
                                ########################

from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)
         
         

# create the classifier and train it
tree = DecisionTreeClassifier(criterion='entropy',max_depth=1 ,random_state=0)
tree.fit(X_train_std,y_train)
                             
y_pred = tree.predict(X_test_std)                                    
                                
print('DECISION TREE LEARNING')
print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))                             
        

y_train_pred = tree.predict(X_train_std) 
print('Training Accuracy: %.3f' % accuracy_score(y_train, y_train_pred))

                        
X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = tree.predict(X_combined)
print('Combined Accuracy: %.3f' % \
        accuracy_score(y_combined, y_combined_pred))








                                ###############
                                #RANDOM FOREST#
                                ###############
from sklearn.ensemble import RandomForestClassifier   

X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)
         
         


forest = RandomForestClassifier(criterion='entropy', n_estimators=12, \
                                    random_state=0, n_jobs=1)
forest.fit(X_train_std,y_train)

y_pred = forest.predict(X_test_std)         # see how we do on the test data

print("RANDOM FOREST")
print('Test Accuracy: %.3f \n' % accuracy_score(y_test, y_pred))

y_train_pred = forest.predict(X_train_std) 
print('Training Accuracy: %.3f' % accuracy_score(y_train, y_train_pred))

                                
X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = forest.predict(X_combined)
print('Combined Accuracy: %.3f' % \
        accuracy_score(y_combined, y_combined_pred))                                
                                
                                
                               ####################
                               #K-NEAREST NEIGHBOR#
                               ####################
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)
         
         


knn = KNeighborsClassifier(n_neighbors=11,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)

y_pred = knn.predict(X_test_std)

print("K-NEAREST NEIGHBOR")
print('Test Accuracy: %.3f \n' % accuracy_score(y_test, y_pred))

y_train_pred = knn.predict(X_train_std) 
print('Training Accuracy: %.3f' % accuracy_score(y_train, y_train_pred))

X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = knn.predict(X_combined)
print('Combined Accuracy: %.3f' % \
        accuracy_score(y_combined, y_combined_pred))





                                