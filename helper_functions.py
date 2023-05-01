import numpy as np


def leaf_prob(y, node_indices):
    """
    The function calculates the probabilities of the node, where the sum of the two probabilities is equal to one.

    Args:
        y (ndarray): Numpy array indicating whether each example at a node is positive (`1`) or negative (`0`)
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
    """
    if node_indices.size == 0:
        return np.array([[1, 0]])

    prob_pos = sum(y[node_indices] == 1) / len(y[node_indices])

    return np.array([[1-prob_pos, prob_pos]])


def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature value to split the node data

    Args:
        X (ndarray): Data matrix of shape(n_samples, n_features)
        y (array like): list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
    """

    best_feature = np.array([-1])
    max_info_gain = np.array([-1])

    for i in range(X.shape[1]):
        temp_info_gain = compute_information_gain(X, y, node_indices, i)
        if temp_info_gain > max_info_gain:
            max_info_gain = temp_info_gain
            best_feature = i

    if max_info_gain == 0:
        return None
    else:
        return best_feature


def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information gain of splitting the node on a given feature

    Args:
        X (ndarray): Data matrix of shape(n_samples, n_features).
        y (array like): list or ndarray with n_samples containing the target variable.
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        feature (int): Index of feature to split on.
    """
    node_indices = node_indices.tolist()
    entropy_node = compute_entropy(y[node_indices])
    left_ind, right_ind = split_dataset(X, node_indices, feature)
    entropy_left = compute_entropy(y[left_ind])
    entropy_right = compute_entropy(y[right_ind])
    w_left = len(left_ind) / len(node_indices)
    w_right = len(right_ind) / len(node_indices)

    cost = entropy_node - (w_left * entropy_left + w_right * entropy_right)

    return cost


def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into left and right branches

    Args:
        X (ndarray): Data matrix of shape(n_samples, n_features).
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered at this step.
        feature (int): Index of feature to split on.
    """

    left_indices_all_input = np.where(X[:, feature] == 1)[0]
    left_indices = list(set(left_indices_all_input).intersection(node_indices))
    left_indices.sort()
    right_indices_all_input = np.where(X[:, feature] == 0)[0]
    right_indices = list(set(right_indices_all_input).intersection(node_indices))
    right_indices.sort()

    return left_indices, right_indices


def compute_entropy(y):
    """
    Computes the entropy

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           positive (`1`) or negative (`0`)
     """

    if len(y) == 0 or np.sum(y) == 0 or np.sum(y) == np.size(y):
        entropy = float(0)
    else:
        p1 = np.sum(y) / np.size(y)
        entropy = float(-p1 * np.log2(p1) - (1-p1) * np.log2(1-p1))

    return entropy
