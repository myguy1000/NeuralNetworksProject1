import numpy as np

class LogisticRegression:
    def __init__(self, input_size, min, max, threshold):
        # Sets the weights and bias to be in the range of -1 to 1.
        # For our standard tests min = -1 and max = 1
        self.weights = np.random.uniform(min, max, input_size)
        self.bias = np.random.uniform(min, max)
        # set threshold to be the threshold value passed in from main
        self.threshold = threshold

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        linear = np.dot(inputs, self.weights) + self.bias
        pred = self.sigmoid(linear)
        #use threshold to determine Label
        if pred > self.threshold:
            return 1
        else:
            return 0

    # Training the model using gradient descent
    def fit(self, inputs, outputs, learning_rate, num_epochs):

        # Gradient descent
        for _ in range(num_epochs):
            #Linear Regression: w.T * X + b
            linear_model = np.dot(inputs, self.weights) + self.bias
            #Use sigmoid to get Logistic regression
            pred = self.sigmoid(linear_model)

            #Compute the gradients
            weight_update = (1 / len(inputs)) * np.dot(inputs.T, (pred - outputs))
            bias_update = (1 /  len(inputs)) * np.sum(pred - outputs)

            #Update weights and bias
            self.weights -= learning_rate * weight_update
            self.bias -= learning_rate * bias_update
