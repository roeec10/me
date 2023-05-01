import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from helper_functions import leaf_prob, get_best_split, split_dataset


class Node:
    """
     A class representing a node in a decision tree.

     “Parameters”:
        feature (int): Index of the feature used to split the data at this node.
        left (Node): The left child node.
        right (Node): The right child node.
        probability (ndarray): The predicted probability of the positive class at this node.
    """
    def __init__(self, feature=None, left=None, right=None, probability=None):
        # Constructor
        self.feature = feature
        self.left = left
        self.right = right
        self.probability = probability

    def is_leaf_node(self):
        return self.probability is not None


class MyID3(BaseEstimator, ClassifierMixin):

    def __init__(self, max_depth=None, root=None, tree_depth=0):
        """
        Constructor for the MyID3 class that build a decision tree.

        “Parameters”:
            max_depth (int or None): The maximum depth of the decision tree. If None, the depth is unlimited.
            root (Node object or None): The root node of the decision tree.
            tree_depth (int) : The maximum depth of the tree.
         """
        self.max_depth = max_depth
        self.root = root
        self.tree_depth = tree_depth

    def fit(self, X, y):
        """
        This method builds the decision tree.

        Args:
            X (ndarray): Data matrix of shape(n_samples, n_features)
            y (array like): list or ndarray with n_samples containing the target variable
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Create an array with indices for each data point
        node_indices = np.arange(len(y))
        # Build the tree using the root node and the array of node indices
        self.root = self.build_tree(X, y, node_indices)

        return self

    def build_tree(self, X, y, node_indices, depth=0):
        """
        This method is used to recursively build the decision tree.

        Args:
            X (ndarray): Data matrix of shape(n_samples, n_features).
            y (array like): list or ndarray with n_samples containing the target variable.
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            depth (int): The current depth of the tree
        """

        # Stopping Criteria: Stops building the tree if any of the following conditions is met:
        # - The maximum depth (self.max_depth) is reached.
        # - There are no more samples in the current node.
        # - All samples in the current node have the same target variable.
        if self.max_depth is None:
            if node_indices.size == 0 or np.all(y[node_indices] == y[node_indices[0]]):
                leaf_decision = leaf_prob(y, node_indices)
                if depth > self.tree_depth:
                    self.tree_depth = depth
                return Node(probability=leaf_decision)
        elif depth >= self.max_depth or node_indices.size == 0 or np.all(y[node_indices] == y[node_indices[0]]):
            leaf_decision = leaf_prob(y, node_indices)
            if depth > self.tree_depth:
                self.tree_depth = depth
            return Node(probability=leaf_decision)

        # Greedy Search: The best feature is selected to split the current node. A new decision node is created with the
        # best feature and the two subtrees created recursively by splitting the current node along the best feature.
        best_feature = get_best_split(X, y, node_indices)
        # Stopping Criteria: If there is no feature that improves the split, a leaf node is created and returned.
        if best_feature is None:
            leaf_decision = leaf_prob(y, node_indices)
            if depth > self.tree_depth:
                self.tree_depth = depth
            return Node(probability=leaf_decision)
        left_ind, right_ind = split_dataset(X, node_indices, best_feature)
        left_subtree = self.build_tree(X, y, np.array(left_ind), depth + 1)
        right_subtree = self.build_tree(X, y, np.array(right_ind), depth + 1)
        return Node(best_feature, left_subtree, right_subtree)

    def predict_single_proba(self, x):
        """
        This method predicts the probability of each class label for a single data point.
        Args:
            x(ndarray): Data matrix of shape(n_samples, n_features).
        """
        # Check if tree exists
        if self.root is None:
            return None

        # Traverse the tree until a leaf node is reached
        node = self.root
        while not node.is_leaf_node():
            if x[node.feature] == 1:
                node = node.left
            else:
                node = node.right

        # Return the predicts probability of each class label
        return node.probability

    def predict_proba(self, X):
        """
        This method predicts the probability of target variable of given data using the fitted decision tree.
        Args:
            X(ndarray): Data matrix of shape(n_samples, n_features).
        """

        # Input validation
        check_array(X)

        # Initialize an empty array to store the probabilities for each sample
        prob = np.empty((0, 2))

        # Iterate over each sample in the input data and predict its probability using the `predict_single_proba` method
        for x in X:
            # Get the probability for the current sample using the `predict_single_proba` method
            sing_prob = self.predict_single_proba(x)
            # Concatenate the resulting probability to `Prob` array
            prob = np.concatenate((prob, sing_prob), axis=0)

        return prob

    def predict(self, X):
        """
        This method predicts the target variable of given data using the fitted decision tree.
        Args:
            X(ndarray): Data matrix of shape(n_samples, n_features).
        """

        # Input validation
        check_array(X)

        # Get the probability of each class for all samples using the `predict_proba` method.
        classes_prob = self.predict_proba(X)
        # Get the index of class with the highest probability for each sample.
        y = np.argmax(classes_prob, axis=1)
        return y
