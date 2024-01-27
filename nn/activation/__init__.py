import numpy as np

def relu(x):
    return np.maximum(0, x)

# Example usage
input_data = np.array([-2, -1, 0, 1, 2])
output = relu(input_data)

print("Input:", input_data)
print("Output:", output)
