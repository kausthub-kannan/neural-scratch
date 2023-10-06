import numpy as np
import scipy.stats as stats


class NaiveBayes:
    def fit(self, x, y):
        """
        Fit the Naive Bayes model to the training data.

        Parameters:
        x (numpy.ndarray): The training data.
        y (numpy.ndarray): The corresponding class labels.

        Compute the mean, variance, and priors for each class.

        >>> X_train = np.array([[1.2, 2.3], [4.5, 5.6], [7.8, 8.9]])
        >>> y_train = np.array([0, 1, 0])
        >>> nb = NaiveBayes()
        >>> nb.fit(X_train, y_train)
        """
        n_samples, n_features = x.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Mean, Variance
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, i_class in enumerate(self._classes):
            x_class = x[y == i_class]
            self._mean[idx, :] = x_class.mean(axis=0)
            self._variance[idx, :] = x_class.var(axis=0)
            self._priors[idx] = x_class.shape[0] / float(n_samples)

    def predict(self, x):
        """
        Predict class labels for the test data.

        Parameters:
        x (numpy.ndarray): The test data.

        Returns:
        numpy.ndarray: The predicted class labels.

        >>> X_test = np.array([[4.5, 5.0], [1.2, 2.3]])
        >>> nb.predict(X_test)
        array([1, 0])
        """
        y_pred = [self._predict(i) for i in x]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        for idx, i_class in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._variance[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


if __name__ == "__main__":
    # Example usage
    X_train = np.array([[1.2, 2.3], [4.5, 5.6], [7.8, 8.9]])
    y_train = np.array([0, 1, 0])

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    X_test = np.array([[4.5, 5.0], [1.2, 2.3]])
    predictions = nb.predict(X_test)

    print("Predictions:")
    print(predictions)
