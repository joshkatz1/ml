from collections import Counter
from typing import Type
from dataclasses import dataclass
import numpy as np


def entropy(label_vector):
    hist = np.bincount(label_vector)
    total_labels = len(label_vector)
    ps = hist / total_labels
    return - np.sum([p * np.log2(p) for p in ps if p >0])


@dataclass
class Node:
    feature: int
    threshold: float
    value: float
    right: object
    left: object

    def is_leaf_node(self):
        return self.value is not None


@dataclass
class DecisionTree:
    n_feature: int = None
    root: object = None
    min_sample_split: int = 2
    max_depth: int = 10

    def fit(self, X, y):
        self.n_feature = X.shape[1]
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feature = X.shape
        n_labels = len(np.unique(y))
        if (depth > self.max_depth or n_samples < self.min_sample_split or n_labels == 1):
            leaf_value = self._m_commen(y)
            return Node(feature=None, threshold=None, value=leaf_value, right=None, left=None)

        feature_idx = np.random.choice(n_feature, self.n_feature, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_idx)
        print(best_feature,best_threshold)
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idx,:], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx,:], y[right_idx], depth + 1)
        return Node(feature=best_feature,threshold=best_threshold,value=None, right=right, left=left)



    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] > node.threshold:
            return self._traverse_tree(x, node.right)
        return self._traverse_tree(x, node.left)

    def _m_commen(self, X):
        occurence_count = Counter(X)
        return occurence_count.most_common(1)[0][0]

    def _best_split(self, X, y, feature_idx):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feature in feature_idx:
            X_cloumn = X[:, feature]
            thresholds = np.unique(X_cloumn)
            for threshold in thresholds:
                gain = self._information_gain(y,X_cloumn, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = threshold
        return split_idx, split_thresh


    def _information_gain(self, y, X_cloumn, threshold):
        parent_entropy = entropy(y)
        left_idx, right_idx = self._split(X_cloumn, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        n_l ,n_r =  len(left_idx), len(right_idx)
        e_l, e_r = entropy(y[left_idx]), entropy(y[right_idx])
        n = len(y)
        child_entropy = (n_l / n) * e_l + (n_r/n) * e_r
        information_gain = parent_entropy - child_entropy
        print(information_gain)
        if information_gain == 0.003184087900009014:
            print(1)
        return information_gain


    def _split(self, X_cloumn, threshold):
        left_idx = np.argwhere(X_cloumn <= threshold).flatten()
        right_idx = np.argwhere(X_cloumn > threshold).flatten()
        return  left_idx, right_idx


if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree(max_depth=15)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(acc)