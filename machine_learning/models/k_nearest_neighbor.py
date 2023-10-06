"""
K-Nearest Neighbors (KNN) is a supervised learning algorithm used for
classification and regression tasks. It works by finding the k-nearest
data points in the training set to a given test point and making predictions
based on the majority class (for classification) or the average value (for regression)
of those neighbors.

Data: The data used for KNN consists of a set of data points, each with a set
of features. In this example, we assume a dataset with 4 features.

Reference: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

Parameters:
n_neighbors (int): The number of neighbors required for KNN.

Create a KNN object with the given number of neighbors.
"""

import numpy as np
from collections import Counter


def euclidean_distance(v1, v2):
    """
    Calculate the Euclidean distance between two vectors.

    Parameters:
    v1 (numpy.ndarray): The first vector.
    v2 (numpy.ndarray): The second vector.

    Returns:
    float: The Euclidean distance between v1 and v2.
    """
    distance = np.sqrt(np.sum((v1 - v2) ** 2))
    return distance


class KNN:
    def __init__(self, n_neighbors: int = 5):
        """
        Parameters:
        n_neighbors (int): The number of neighbors required for KNN.

        Create a KNN object with the given number of neighbors.
        """
        self.n = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, vector: np.ndarray[float], labels: np.ndarray[int]) -> None:
        """
        Parameters:
        vector (numpy.ndarray): The training data.
        labels (numpy.ndarray): The corresponding labels.

        Fit the KNN model to the training data.

        >>> features_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> target_train = np.array([0, 1, 0, 1])
        >>> knn_test_mode = KNN(n_neighbors=2)
        >>> knn_test_mode.fit(features_train, target_train)
        """
        self.X_train = vector
        self.y_train = labels

    def predict(self, vector: np.ndarray[float]):
        """
        Parameters:
        vector (numpy.ndarray): The test data for prediction.

        Predict labels for the test data using KNN.

        >>> features_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> target_train = np.array([0, 1, 0, 1])
        >>> knn_test_mode = KNN(n_neighbors=2)
        >>> knn_test_mode.fit(features_train, target_train)
        >>> features_test = np.array([[4.5, 5], [1.5, 2.5]])
        >>> knn_test_mode.predict(features_test)
        [1, 0]
        """
        return [self._predict(x) for x in vector]

    def _predict(self, x):
        """
        Calculate the nearest distance and predict the label.

        Parameters:
        x (numpy.ndarray): A single test data point.

        Returns:
        int: The predicted label.
        """
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idxs = np.argsort(distances)[:self.n]
        k_nearest_labels = [self.y_train[idx] for idx in k_idxs]
        y_pred = Counter(k_nearest_labels).most_common(1)[0][0]
        return y_pred


if __name__ == "__main__":
    # Example usage
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 1, 0, 1])

    X_test = np.array([[2.5, 3.5], [1.5, 2.5]])

    knn = KNN(n_neighbors=2)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    print("Predictions:", predictions)
