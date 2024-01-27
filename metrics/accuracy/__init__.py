def accuracy(y_true, y_pred):
    # Ensure the input lists have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists must have the same length.")

    # Count the number of correct predictions
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    # Calculate accuracy
    accuracy = correct_predictions / len(y_true)

    return accuracy

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

acc = accuracy(y_true, y_pred)
print("Accuracy:", acc)
