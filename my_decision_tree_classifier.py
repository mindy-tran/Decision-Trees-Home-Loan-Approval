import numpy as np
from collections import Counter


class MyDecisionTreeClassifier:

    def __init__(self, required_num__split=2, max_tree_depth=5):
        """
        One typically initializes shared class variables and data structures in the constructor.

        Variables which you wish to modify in train(X, y) and then utilize again in predict(X)
        should be explicitly initialized here (even only as self.my_variable = None).
        """
        self.required_num__split = required_num__split
        self.max_tree_depth = max_tree_depth
        self.root = None

    def _build(self, X, y, depth=0):
        '''
        Build a decision tree, recursive

        :param X: np.array, features
        :param y: np.array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: MyNode
        '''
        n_rows, n_cols = X.shape

        # Check to see if a MyNode should be leaf
        if n_rows >= self.required_num__split and depth <= self.max_tree_depth:
            # choose the best split
            best = best_split(X, y)
            # If the split isn't pure
            if best['infogain'] > 0:
                # Build a tree on the left
                left = self._build(
                    X=best['le_threshold'][:, :-1],
                    y=best['le_threshold'][:, -1],
                    depth=depth + 1
                )
                right = self._build(
                    X=best['gt_threshold'][:, :-1],
                    y=best['gt_threshold'][:, -1],
                    depth=depth + 1
                )
                return MyNode(
                    feature=best['feature_index'],
                    threshold=best['threshold'],
                    left_value=left,
                    right_value=right,
                    infogain=best['infogain']
                )
        # Leaf MyNode - value is the most common target value
        return MyNode(
            value=Counter(y).most_common(1)[0][0]
        )

    def fit(self, X, y):
        """
        This is the method which will be called to train the model. We can assume that train will
        only be called one time for the purposes of this project.

        :param X: The samples and features which will be used for training. The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]
        :param y: The target/response variable used for training. The data should have the shape:
        y = [target_for_sample_a, target_for_sample_b, ..., target_for_sample_n]

        :return: self Think of this method not having a return statement at all. The idea to
        "return self" is a convention of scikit learn; the underlying model should have some
        internally saved trained state.
        """
        # Call a recursive function to build the tree
        self.root = self._build(X, y)
        return self

    def _predict(self, x, decision_tree):
        '''
        Predict to go left or right, recursive

        :param x: single observation
        :param decision_tree: built tree
        :return: float, predicted class
        '''
        # Leaf MyNode
        if decision_tree.value != None:
            return decision_tree.value
        feature_value = x[decision_tree.feature]

        # Go to the left
        if feature_value <= decision_tree.threshold:
            return self._predict(x=x, decision_tree=decision_tree.left_value)

        # Go to the right
        if feature_value > decision_tree.threshold:
            return self._predict(x=x, decision_tree=decision_tree.right_value)

    def predict(self, X):
        """
        This is the method which will be used to predict the output targets/responses of a given
        list of samples.

        It should rely on mechanisms saved after train(X, y) was called.
        You can assume that train(X, y) has already been called before this method is invoked for
        the purposes of this project.

        :param X: The samples and features which will be used for prediction. The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]
        :return: The target/response variables the model decides is optimal for the given samples.
        The data should have the shape:
        y = [prediction_for_sample_a, prediction_for_sample_b, ..., prediction_for_sample_n]
        """
        # Call the _predict() function for every observation
        return [self._predict(x, self.root) for x in X]


class MyNode:
    '''
    Helper class which implements a single tree MyNode.
    '''
    def __init__(self, feature=None, threshold=None, left_value=None, right_value=None, infogain=None, value=None):
        self.feature = feature
        self.infogain = infogain
        self.threshold = threshold
        self.value = value
        self.left_value = left_value
        self.right_value = right_value

# Here are more functions I defined outside of the class to do calculations for entropy and info gain
def entropy(y):
    '''
    Calc entropy

    :param y: list, y-values
    :return: float, entropy value
    '''
    # Convert to ints
    # counts occurances of each value in an array
    counts = np.bincount(np.array(y, dtype=np.int64))
    # fraction of each class label
    fraction = counts / len(y)

    # Calc entropy
    entropy = 0
    for fct in fraction:
        if fct > 0:
            entropy += fct * np.log2(fct)
    return -entropy


def info_gain(root, feature1, feature2):
    '''
    Calc information gain
    :param root: list, root MyNode
    :param feature1: list, 1st child of root
    :param feature2: list, 2nd child of root
    :return: float, information gain
    '''
    # IG = H(root) - H(feature)
    H_root = entropy(root)
    count_feature1 = len(feature1) / len(root)
    count_feature2 = len(feature2) / len(root)
    H_feature = (count_feature1 * entropy(feature1) + count_feature2 * entropy(feature2))
    return H_root - H_feature


def best_split(X, y):
    '''
    Calc best split based on info gain

    :param X: np.array, features
    :param y: np.array or list, target
    :return: dict
    '''
    best_split = {}
    best_info_gain = -1
    n_rows, n_cols = X.shape

    # For every dataset feature
    for f_idx in range(n_cols):
        X_curr = X[:, f_idx]
        # look at each feature
        # separate left and right parts
        for threshold in np.unique(X_curr):
            # Left has <= to the threshold
            # Right has records > than the threshold
            df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
            # le = less than or equal to
            le_threshold = np.array([row for row in df if row[f_idx] <= threshold])
            # gt = greater than
            gt_threshold = np.array([row for row in df if row[f_idx] > threshold])

            # Do the calculation only if there's data in both subsets
            if len(le_threshold) > 0 and len(gt_threshold) > 0:
                # Obtain the value of the target variable for subsets
                y = df[:, -1]
                y_left = le_threshold[:, -1]
                y_right = gt_threshold[:, -1]

                # Calc the info gain and save the split parameters
                # if the current split if better then the previous best

                infogain = info_gain(y, y_left, y_right)
                # if the current infogain is higher than our current best, replace the current best with our current infogain
                if infogain > best_info_gain:
                    best_split = {
                        'feature_index': f_idx,
                        'threshold': threshold,
                        'le_threshold': le_threshold,
                        'gt_threshold': gt_threshold,
                        'infogain': infogain
                    }
                    best_info_gain = infogain
    return best_split # the best one based on Greedy, info gain



