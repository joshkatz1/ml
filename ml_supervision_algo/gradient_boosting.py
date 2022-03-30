import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingFromScratch():

    def __init__(self, n_trees, learning_rate, max_depth=1):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, x, y):
        self.trees = []
        self.F0 = y.mean()
        Fm = self.F0
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(x, y - Fm)
            Fm += self.learning_rate * tree.predict(x)
            self.trees.append(tree)

    def predict(self, x):
        return self.F0 + self.learning_rate * np.sum([tree.predict(x) for tree in self.trees], axis=0)


# Testing
def change(y):
    if y < 0:
        return 0
    if y > 1:
        return 1
    if y > 0 and y < 1:
        if y > 0.5:
            return 1
        else:
            return 0


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

    # Adaboost classification with 5 weak classifiers
    clf = GradientBoostingFromScratch(n_trees=50, learning_rate=0.05, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = np.array([change(y) for y in y_pred])
    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)

    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)