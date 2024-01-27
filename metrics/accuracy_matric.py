import numpy as np

def accuracy(y_true, y_pred):
    # Calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)

    # Calculate total number of predictions
    total_predictions = len(y_true)

    # Calculate accuracy
    acc = correct_predictions / total_predictions if total_predictions > 0 else 0

    return acc

# Example usage for demo
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

acc = accuracy(y_true, y_pred)
print("Accuracy:", acc)
