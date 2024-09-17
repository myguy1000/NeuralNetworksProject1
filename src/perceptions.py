import numpy as np

"""
Initialization for perceptron objects
Able to decide the size of inputs, the min and max for the bias and weights, as well as the threshold.
Changing these values
can greatly change the learning of the perceptron
"""
class Perceptron:
    def __init__(self, input_size, min, max, threshold):
        #Sets the weights and bias to be in the range of -1 to 1.
        #For our standard tests min = -1 and max = 1
        self.weights = np.random.uniform(min, max, input_size)
        self.bias = np.random.uniform(min, max)
        #set threshold to be the threshold value passed in from main
        self.threshold = threshold

    """
    getBias returns the bias value for the perceptron
    """
    def getBias(self):
        return self.bias
    """
    The forward method computes the weighted sum of inputs, adds the bias, and 
    compares it to the threshold to make a prediction the output is 1 if it is greater, otherwise it returns a zero
    used in the fit function
    """
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

    """
    fit is the perceptron learning function
    fit re-adjusts the weights and bias for the perceptron when it is trying to better fit the data
    if (curOutput-prediction) > 0 then the bias and weights change, if they are the same the the w and b don't change
    """
    def fit(self, inputs, output, learning_rate, num_epochs):
        # Train the current_perceptron using the current_perceptron learning algorithm
        z = range(num_epochs)
        for i in z:
            curInput = 0
            for x in inputs:
                curOutput = output[curInput]
                prediction = self.forward(x)

                self.weights += learning_rate * (curOutput - prediction) * x
                self.bias += learning_rate * (curOutput - prediction)



                curInput += 1


    def fit_GD(self, inputs, output, learning_rate, num_epochs):
        """
        Trains the perceptron using the gradient descent algorithm with Mean Squared Error (MSE) as the loss function.
        The weights and bias are updated by calculating the partial derivatives (gradients) from the accumulated
        loss over all samples.
        """
        for epoch in range(num_epochs):
            total_loss = 0
            gradient_w = np.zeros_like(self.weights)
            gradient_b = 0

            for i, x in enumerate(inputs):
                y_true = output[i]

                # Use the existing forward method
                prediction = self.forward(x)

                # Calculate the loss using Mean Squared Error (MSE)
                loss = 0.5 * (y_true - prediction) ** 2
                total_loss += loss

                # Calculate gradients
                error = y_true - prediction
                gradient_w += -error * x
                gradient_b += -error

            # Update weights and bias
            self.weights -= learning_rate * gradient_w / len(inputs)
            self.bias -= learning_rate * gradient_b / len(inputs)

            # Print the average loss for this epoch
            #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(inputs)}")

