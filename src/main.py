import numpy as np
import matplotlib.pyplot as plt
from perceptions_old import Perceptron  # Ensure the class is saved as perceptron.py

THRESHOLD = 0.5
LEARNING_RATE = 0.1
NUM_EPOCHS = 10
NUMSAMPLES = 100
MINSAMPLESPACE = -100
MAXSAMPLESPACE = 100


def case1Samples():
    case1Class1 = []
    for i in range(NUMSAMPLES):
        test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)
        case1Class1.append(test)
        # print(test)
    case1Class2 = []
    for i in range(NUMSAMPLES):
        case1Class2.append(np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2))
    trainingDataX = []
    for i in range(NUMSAMPLES):
        trainingDataX.append(case1Class1[i])
        trainingDataX.append(case1Class2[i])
    # for i in range(NUMSAMPLES):
    # print(trainingDataX[i])
    trainingDataY = []
    for i in range(len(trainingDataX)):
        if (-1 * trainingDataX[i][0]) + trainingDataX[i][1] > 0:
            trainingDataY.append(1)
        else:
            trainingDataY.append(0)

    trainingDataX = np.array(trainingDataX)
    trainingDataY = np.array(trainingDataY)
    return trainingDataX, trainingDataY

def case2Samples():
    case2Class1 = []
    for i in range(NUMSAMPLES):
        test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)
        case2Class1.append(test)
        # print(test)
    case2Class2 = []
    for i in range(NUMSAMPLES):
        case2Class2.append(np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2))
    trainingDataX = []
    for i in range(NUMSAMPLES):
        trainingDataX.append(case2Class1[i])
        trainingDataX.append(case2Class2[i])
    # for i in range(NUMSAMPLES):
    # print(trainingDataX[i])
    trainingDataY = []
    for i in range(len(trainingDataX)):
        if (trainingDataX[i][0] - (2 * trainingDataX[i][1]) + 5) > 0:
            trainingDataY.append(1)
        else:
            trainingDataY.append(0)

    trainingDataX = np.array(trainingDataX)
    trainingDataY = np.array(trainingDataY)

    return trainingDataX, trainingDataY

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
    plt.show()

# Case 1
X_train_1, y_train_1 = case1Samples()
perceptron_1 = Perceptron(2,-1,1,THRESHOLD)
perceptron_1.fit(X_train_1, y_train_1, LEARNING_RATE, NUM_EPOCHS)

# Generate test data for Case 1
X_test_1, y_test_1 = case1Samples()
misclassified_1 = incorrectlyClassified(perceptron_1, X_test_1, y_test_1)
print(f"Case 1: Misclassified samples = {misclassified_1}")


# Plot decision boundary for Case 1
makeCurrentPlot(perceptron_1, X_train_1, y_train_1, title='Case 1: Decision Boundary')

# Case 2
X_train_2, y_train_2 = case2Samples()
perceptron_2 = Perceptron(2,-1,1,THRESHOLD)
#plot_decision_boundary(perceptron_2, X_train_2, y_train_2, title='Pre Boundry')
perceptron_2.fit(X_train_2, y_train_2, LEARNING_RATE, NUM_EPOCHS)
makeCurrentPlot(perceptron_2, X_train_2, y_train_2, title='Case 2: Decision Boundary')
# Generate test data for Case 2
X_test_2, y_test_2 = case2Samples()
misclassified_2 = incorrectlyClassified(perceptron_2, X_test_2, y_test_2)
print(f"Case 2: Misclassified samples = {misclassified_2}")



# Plot decision boundary for Case 2
#plot_decision_boundary(perceptron_2, X_train_2, y_train_2, title='Case 2: Decision Boundary')
