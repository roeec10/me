from Bagging_algorithm import MyBaggingID3
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import time


def cross_validation(X, y, estimator=MyBaggingID3(), cv=5, repetitions=1):
    """
    This function is used for performing k-fold cross-validation with a given estimator. It splits the data into
    cv folds, trains and evaluates the estimator repetitions times, and returns the mean
    performance score across all repetitions.

    “Parameters”:
        X (ndarray): Data matrix of shape(n_samples, n_features)
        y (array like): list or ndarray with n_samples containing the target variable
        estimator (object): The estimator object to use for training and evaluating.
        cv (int): The number of folds to split the data into for cross-validation.
        repetitions (int): The number of times to repeat the cross-validation process.
    """

    # calculate important parameters
    num_samples = X.shape[0]
    fold_size = int(np.ceil(len(y)/cv))
    # evaluation metric
    score_dict = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], 'fit_runtime': []}
    # Iterative over the k-fold cross validation n repetitions
    for n in range(repetitions):
        # Shuffle the Data before split to k folds
        ind_shuffle = np.random.choice(num_samples, size=num_samples, replace=False)
        # find the indices of each k-fold
        k_fold_ind_list = []
        for i in range(cv):
            k_fold_ind_list.append(ind_shuffle[i*fold_size: (i+1) * fold_size])

        # Train the model k times and evaluate the performance
        for i in range(cv):
            # Split data into train and test sets
            X_train = np.delete(X, k_fold_ind_list[i], axis=0)
            y_train = np.delete(y, k_fold_ind_list[i])
            X_test = X[k_fold_ind_list[i]]
            y_test = y[k_fold_ind_list[i]]

            # Fit the estimator and make predictions
            start_time = time.time()
            Bagging_ensemble = estimator
            Bagging_ensemble.fit(X_train, y_train)
            end_time = time.time()
            y_prediction = Bagging_ensemble.predict(X_test)
            y_prediction_proba = Bagging_ensemble.predict_proba(X_test)

            # Record evaluation metrics for this iteration
            score_dict['accuracy'].append(accuracy_score(y_test, y_prediction))
            score_dict['precision'].append(precision_score(y_test, y_prediction))
            score_dict['recall'].append(recall_score(y_test, y_prediction))
            score_dict['f1'].append(f1_score(y_test, y_prediction))
            score_dict['roc_auc'].append(roc_auc_score(y_test, y_prediction_proba[:, 1]))
            score_dict['fit_runtime'].append(end_time - start_time)

    # Calculate mean performance scores across all iterations
    score_pd = pd.DataFrame(score_dict)
    mean_score = score_pd.mean()

    return mean_score
