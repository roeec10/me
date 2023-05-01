import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from ID3_algorithm import MyID3


class MyBaggingID3(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_samples=float(1), max_features=float(1),
                 max_depth=None, forest=None, base_estimator=MyID3, classes_=None, feature_indices=None):
        """
        Constructor for the MyBaggingID3 class that builds bagging ensemble using the MyID3 as base classifier and
        bagging technique.

        “Parameters”:
        n_estimators (int): The number of decision trees in the bagging ensemble.
        max_samples (float): The fraction of the training samples to draw from the training data with replacement.
        max_features (float): The fraction of the total number of features to use when building each decision tree.
        max_depth (int or None): The maximum depth of each decision tree. If None, the depth is unlimited.
        forest (list) : The list of decision trees that comprise the ensemble.
        base_estimator (class): The base estimator used to build each decision tree in the bagging ensemble. Must have
                                a `fit`, `predict`, and `predict_proba` method.
        classes_ (array-like) : The class labels.
        feature_indices (ndarray) : The indices of the features used for each tree.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_depth = max_depth
        self.forest = forest
        self.base_estimator = base_estimator
        self.classes_ = classes_
        self.feature_indices = feature_indices

    def fit(self, X, y):
        """
        This method builds a forest of decision trees using bagging.

        Args:
            X (ndarray): Data matrix of shape(n_samples, n_features)
            y (array like): list or ndarray with n_samples containing the target variable
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Determine the number of samples and features for the trees based on max_samples and max_features  .
        num_samples = int(np.floor(self.max_samples * X.shape[0]))
        num_features = int(np.floor(self.max_features * X.shape[1]))
        # Initialize the trees list and the feature_indices.
        self.forest = []
        init_feature_indices_matrix = np.zeros((self.n_estimators, num_features))
        self.feature_indices = init_feature_indices_matrix.astype(int)
        # Iterate over the number of trees to build
        for i in range(self.n_estimators):
            # Randomly select `num_samples` samples with replacement
            draw_sample_ind = np.sort(np.random.choice(X.shape[0], size=num_samples, replace=True))
            # Randomly select `num_features` features without replacement
            draw_features_ind = np.sort(np.random.choice(X.shape[1], size=num_features, replace=False))
            # Store the indices of the features that selected randomly for this specific tree.
            self.feature_indices[i, :] = draw_features_ind
            # Select the samples and features from the original dataset
            x_random = X[draw_sample_ind][:, draw_features_ind]
            y_random = y[draw_sample_ind]
            # Create a new instance of the base estimator and fit it to the randomly selected dataset
            single_tree = deepcopy(self.base_estimator(self.max_depth))
            single_tree.fit(x_random, y_random)
            # Add the fitted tree to the forest
            self.forest.append(single_tree)

        return self

    def predict(self, X):
        """
        Predicts the target variable of the given data using the fitted forest of decision trees.
        Args:
            X(ndarray): Data matrix of shape(n_samples, n_features).
        """

        # Input validation
        check_array(X)

        # Create an empty matrix to store predictions from each tree
        prediction_mat = np.zeros((len(X), len(self.forest)))
        # for loop to predict the target variable using each tree in the forest
        for i, single_tree in enumerate(self.forest):
            prediction_mat[:, i] = single_tree.predict(X[:, self.feature_indices[i]])

        # Use the majority vote concept to get the prediction
        y = np.squeeze(np.round(np.mean(prediction_mat, axis=1)))
        return y.astype(int)

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given data using the fitted forest of decision trees.
        Args:
            X(ndarray): Data matrix of shape(n_samples, n_features).
        """

        # Input validation
        check_array(X)

        # Create an empty matrix to store predicted probabilities from each tree
        prob = np.zeros((len(X), 2))
        # for loop to predict probabilities using each tree in the forest
        for i, single_tree in enumerate(self.forest):
            prob += single_tree.predict_proba(X[:, self.feature_indices[i]])

        # calculate the average of probabilities from all trees
        prob = prob / len(self.forest)
        return prob
