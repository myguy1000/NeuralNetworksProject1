import numpy as np


class Perceptron:
    def __init__(self, input_size, min, max, threshold):
        # Initialize weights and bias to random numbers in the range of [-1, 1]
        self.weights = np.random.uniform(min, max, input_size)
        self.bias = np.random.uniform(min, max)
        self.threshold = threshold
    def getBias(self):
        return self.bias

    def forward(self, inputs):
        # Compute the prediction for the class of each input of the current_perceptron
        curItem = 0
        total = 0
        for i in inputs:
            total = total + (i * self.weights[curItem])
            curItem += 1

        total = total + self.bias

        if total > self.threshold:
            return 1
        else:
            return 0

    def fit(self, inputs, output, learning_rate, num_epochs):
        # Train the current_perceptron using the current_perceptron learning algorithm
        z = range(num_epochs)
        for i in z:
            curInput = 0
            for x in inputs:
                curOutput = output[curInput]
                prediction = self.forward(x)
                #print("Output: " + str(curOutput))
                #print("prediction " + str(prediction))
                #print("x: " + str(x))
                #print("preweight: " + str(self.weights))
                #print("prebias: " + str(self.bias))
                self.weights += learning_rate * (curOutput - prediction) * x
                self.bias += learning_rate * (curOutput - prediction)
                #print("postweight: " + str(self.weights))
                #print("postbias: " + str(self.bias))
                #print("---------------------")
                #print("---------------------")


                curInput += 1
