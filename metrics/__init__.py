import numpy as np

def f1_score(y_true, y_pred):
    """
    Compute the F1 score.

    Parameters:
    - y_true: Array of true labels.
    - y_pred: Array of predicted labels.

    Returns:
    - F1 score.
    """
    # Convert to NumPy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate True Positives, False Positives, False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate precision and recall, avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1 score, avoid division by zero
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1

# Test the function
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0, 1]
print("F1 Score:", f1_score(y_true, y_pred))
