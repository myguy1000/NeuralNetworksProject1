import numpy as np
import matplotlib.pyplot as plt
from perceptions import Perceptron  # Ensure the class is saved as perceptron.py

THRESHOLD = 0.5
LEARNING_RATE = 0.1
NUM_EPOCHS = 10
NUMSAMPLES = 100
MINSAMPLESPACE = -100
MAXSAMPLESPACE = 100

# Function: caseSamples
# Description: generates samples for each of the three test cases
# input: an integer reflecting the current case to make samples for ([1,2,3])
# output: X_shuffled and y_shuffled -- the randomly generated values for the X features and the associated, actual y attribute
def caseSamples(test_case):

    # Initialize empty lists to store valid samples for class 1 and class 2
    class1_samples = []
    class2_samples = []
    # iterate twice: once to make class1 samples, once to make class2 samples
    for i in range(2):
        class_samples = []
        # Generate samples for each class until we have exactly 100 valid samples
        while len(class_samples) < 100:
            # Generate random values for X1, X2, X3, X4
            X1 = np.random.uniform(-10, 10)
            X2 = np.random.uniform(-10, 10)
            X3 = np.random.uniform(-10, 10)
            X4 = np.random.uniform(-10, 10)
            
            test = None
            if test_case==1:
                if i == 0:
                    test = (X2 - X1 > 0)
                else:
                    test = (X2 - X1 < 0)
            elif test_case==2:
                if i == 0:
                    test = X1 - 2 * X2 + 5 > 0
                else:
                    test = X1 - 2 * X2 + 5 < 0

            else: # i.e. test_case ==3
                if i == 0:
                    test = (0.5 * X1 - X2 - 10 * X3 + X4 + 50 > 0)
                else:
                    test = (0.5 * X1 - X2 - 10 * X3 + X4 + 50 < 0)

            # Check if the sample satisfies the condition for class 1
            if test:
                if test_case != 3:
                    class_samples.append([X1, X2])
                else:
                    class_samples.append([X1, X2, X3, X4])

        if i == 0:
            # Convert class1_samples to a NumPy array
            class1_samples = np.array(class_samples)
        else:
            class2_samples = np.array(class_samples)

    # Combine both classes
    X = np.vstack((class1_samples, class2_samples))

    # Create labels for the classes (1 for class 1, 0 for class 2)
    y = np.concatenate([np.ones(100), np.zeros(100)])

    # Combine X and y into a single array where y is appended as the last column of X
    combined = np.hstack((X, y.reshape(-1, 1)))

    # Shuffle the combined array along the rows
    np.random.shuffle(combined)

    # Split the combined array back into X and y
    X_shuffled = combined[:, :-1]  # All columns except the last one are X
    y_shuffled = combined[:, -1]   # The last column is y

    return X_shuffled, y_shuffled

def incorrectlyClassified(perceptron, xInputs, yInputs):
    incorrect = 0
    for i in range(len(xInputs)):
        nextPrediction = perceptron.forward(xInputs[i])
        correct = yInputs[i]
        if nextPrediction != correct:
            incorrect = incorrect + 1
        else:
            incorrect = incorrect + 0
    return incorrect

def makeCurrentPlot(perceptron, xValues, yValues, title):
    # Plot the data points
    class_1 = xValues[yValues == 1]
    class_2 = xValues[yValues == 0]
    graphX = []
    graphY = []
    for i in class_1:
        graphX.append(i[0])
    for i in class_1:
        graphY.append(i[1])

    graphX2 = []
    graphY2 = []
    for i in class_2:
        graphX2.append(i[0])
    for i in class_2:
        graphY2.append(i[1])

    plt.scatter(graphX, graphY, color='b', label='Class 1')
    plt.scatter(graphX2, graphY2, color='r', label='Class 2')

    # Plot the decision boundary: w1*x1 + w2*x2 + bias = 0 => x2 = -(w1/w2)*x1 - (bias/w2)
    x_values = np.linspace(MINSAMPLESPACE, MAXSAMPLESPACE, 100)
    w1, w2 = perceptron.weights
    bias = perceptron.bias
    y_values = -(w1/w2) * x_values - (bias/w2)
    plt.plot(x_values, y_values, 'g--', label='Decision Boundary')

    # Plot details
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.xlim(-12.5,12.5)
    plt.ylim(-12.5,12.5)
    plt.show()

# Case 1
X_train_1, y_train_1 = caseSamples(1)
perceptron_1 = Perceptron(2,-1,1,THRESHOLD)
perceptron_1.fit(X_train_1, y_train_1, LEARNING_RATE, NUM_EPOCHS)

# Generate test data for Case 1
X_test_1, y_test_1 = caseSamples(1)
misclassified_1 = incorrectlyClassified(perceptron_1, X_test_1, y_test_1)
print(f"Case 1: Misclassified samples = {misclassified_1}")


# Plot decision boundary for Case 1
makeCurrentPlot(perceptron_1, X_train_1, y_train_1, 'Case 1: Decision Boundary')

# Case 2
X_train_2, y_train_2 = caseSamples(2)
perceptron_2 = Perceptron(2,-1,1,THRESHOLD)
#plot_decision_boundary(perceptron_2, X_train_2, y_train_2, title='Pre Boundry')
perceptron_2.fit(X_train_2, y_train_2, LEARNING_RATE, NUM_EPOCHS)
makeCurrentPlot(perceptron_2, X_train_2, y_train_2, 'Case 2: Decision Boundary')
# Generate test data for Case 2
X_test_2, y_test_2 = caseSamples(2)
misclassified_2 = incorrectlyClassified(perceptron_2, X_test_2, y_test_2)
print(f"Case 2: Misclassified samples = {misclassified_2}")

# Case 3

X_train_3, y_train_3 = caseSamples(3)
perceptron_3 = Perceptron(4,-1,1,THRESHOLD)

# regular fit function
perceptron_3.fit(X_train_3, y_train_3, LEARNING_RATE, NUM_EPOCHS)
misclassified_3 = incorrectlyClassified(perceptron_3, X_train_3, y_train_3)
print(f"Case 3: Misclassified samples (train) = {misclassified_3}")

# GD fit function
perceptron_3.fit_GD(X_train_3, y_train_3, LEARNING_RATE, NUM_EPOCHS)
misclassified_3 = incorrectlyClassified(perceptron_3, X_train_3, y_train_3)
print(f"Case 3: Misclassified samples (train) = {misclassified_3}")

# Generate test data for Case 2
X_test_3, y_test_3 = caseSamples(3)
misclassified_3 = incorrectlyClassified(perceptron_3, X_test_3, y_test_3)
print(f"Case 3: Misclassified samples (test) = {misclassified_3}")
