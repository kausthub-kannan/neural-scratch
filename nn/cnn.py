import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target'].astype(np.int)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = OneHotEncoder(sparse=False, categories='auto')
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

class ConvLayer:
    def __init__(self, input_channels, num_filters, filter_size, padding=0, stride=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size)
        self.bias = np.zeros((num_filters, 1))
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        self.input = input
        h, w = input.shape
        output_h = (h - self.filter_size + 2 * self.padding) // self.stride + 1
        output_w = (w - self.filter_size + 2 * self.padding) // self.stride + 1
        output = np.zeros((self.num_filters, output_h, output_w))

        padded_input = np.pad(input, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for f in range(self.num_filters):
            for i in range(0, h - self.filter_size + 1, self.stride):
                for j in range(0, w - self.filter_size + 1, self.stride):
                    im_region = padded_input[i:i + self.filter_size, j:j + self.filter_size]
                    output[f, i // self.stride, j // self.stride] = np.sum(im_region * self.filters[f]) + self.bias[f]

        return output

class MaxPoolLayer:
    def __init__(self, filter_size=2, stride=2):
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        h, w = input.shape
        output_h = h // self.stride
        output_w = w // self.stride
        output = np.zeros((output_h, output_w))

        for i in range(0, h, self.stride):
            for j in range(0, w, self.stride):
                im_region = input[i:(i + self.filter_size), j:(j + self.filter_size)]
                output[i // self.stride, j // self.stride] = np.max(im_region)

        return output

class FlattenLayer:
    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten()

class FullyConnectedLayer:
    def __init__(self, num_inputs, num_outputs):
        self.weights = np.random.randn(num_outputs, num_inputs)
        self.bias = np.zeros((num_outputs, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

learning_rate = 0.01
epochs = 10
batch_size = 64

conv1 = ConvLayer(input_channels=1, num_filters=8, filter_size=3, padding=1, stride=1)
pool1 = MaxPoolLayer(filter_size=2, stride=2)
flatten = FlattenLayer()
fc1 = FullyConnectedLayer(14 * 14 * 8, 128)
fc2 = FullyConnectedLayer(128, 10)

for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train_onehot[i:i + batch_size]

        conv_output = conv1.forward(batch_X.reshape(-1, 28, 28))
        activation1 = relu(conv_output)
        pool_output = pool1.forward(activation1)
        flattened = flatten.forward(pool_output)
        fc1_output = fc1.forward(flattened)
        activation2 = relu(fc1_output)
        scores = fc2.forward(activation2)
        probs = softmax(scores)

        delta = probs - batch_y
        d_fc2 = np.dot(delta.T, activation2)
        delta = np.dot(delta, fc2.weights)
        delta = delta.reshape(pool_output.shape)
        delta = relu_derivative(pool_output) * delta
        d_pool = delta
        delta = conv1.backprop(delta)
        
        fc2.weights -= learning_rate * d_fc2.T
        fc2.bias -= learning_rate * np.sum(delta, axis=(0, 1, 2), keepdims=True)
        
        if i % (batch_size * 10) == 0:
            loss = cross_entropy_loss(np.argmax(batch_y, axis=1), probs)
            print(f'Epoch {epoch + 1}, Iteration {i}: Loss {loss}')

conv_output = conv1.forward(X_test.reshape(-1, 28, 28))
activation1 = relu(conv_output)
pool_output = pool1.forward(activation1)
flattened = flatten.forward(pool_output)
fc1_output = fc1.forward(flattened)
activation2 = relu(fc1_output)
scores = fc2.forward(activation2)
probs = softmax(scores)
predictions = np.argmax(probs, axis=1)
accuracy = np.mean(predictions == y_test)
print(f'Test accuracy: {accuracy}')