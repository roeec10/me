from Pre_procesing import chose_dataset
from cross_validation_test import cross_validation
from Bagging_algorithm import MyBaggingID3
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

"""
Description : This code allows to perform bagging-based classification on a dataset of your choice
from the provided pull, by using the pre-processing, bagging algorithms provided and cross-validation.
- First, you select the dataset to use from the options provided.
- Second,you define the classifier to use for bagging and set its hyperparameters
- Last, you set the hyperparameters for the cross validation.
"""

# Specify the dataset to use by setting the variable 'dataset' to one of the options
# (e.g. 'car', 'monks', 'mushrooms', 'tic-tac-toe', 'spect')
dataset = 'car'
# Load the dataset and perform pre-processing
X_data, y_data = chose_dataset(dataset)

"""
Define the Classifier (MyBaggingID3() or BaggingClassifier())  and is hyperparameters: "n_estimators", "max_samples ",
"max_features", "Max_depth". In the sklearn BaggingClassifier() the "max_depth" need to be define in the base estimator
that in our case is DecisionTreeClassifier().
"""
estimator = MyBaggingID3(n_estimators=200, max_samples=0.7, max_features=0.9)

"""
Evaluate the predictive performance of the classifier using k-fold cross validation.
 Specify:
- estimator: the classifier to use
- cv: the number of folds to split the data into for cross validation
- repetitions: the number of times to repeat the cross validation process

The function returns the mean score for each test of the cross validation.
"""
mean_score = cross_validation(X_data, y_data, estimator=estimator, cv=5, repetitions=2)
