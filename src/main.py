import numpy as np
import matplotlib.pyplot as plt
from perceptions import Perceptron  # Ensure the class is saved as perceptron.py

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
    #print("number of instances > 0: ", np.count_nonzero(trainingDataY == 1))
    #print("number of instances < 0: ", np.count_nonzero(trainingDataY == 0))
    return trainingDataX, trainingDataY

def case3Samples():

    # Initialize empty lists to store valid samples for class 1 and class 2
    class1_samples = []
    count = 0
    # Generate samples for class 1 until we have exactly 100 valid samples
    while len(class1_samples) < 100:
        # Generate random values for X1, X2, X3, X4
        X1 = np.random.uniform(-10, 10)
        X2 = np.random.uniform(-10, 10)
        X3 = np.random.uniform(-10, 10)
        X4 = np.random.uniform(-10, 10)
        
        # Check if the sample satisfies the condition for class 1
        count += 1
        if 0.5 * X1 - X2 - 10 * X3 + X4 + 50 > 0:
            class1_samples.append([X1, X2, X3, X4])

   # Convert class1_samples to a NumPy array
    class1_samples = np.array(class1_samples)

    # Now generate 100 samples for class 2 where 0.5*X1 - X2 - 10*X3 + X4 + 50 < 0
    class2_samples = []

    while len(class2_samples) < 100:
        # Generate random values for X1, X2, X3, X4
        X1 = np.random.uniform(-10, 10)
        X2 = np.random.uniform(-10, 10)
        X3 = np.random.uniform(-10, 10)
        X4 = np.random.uniform(-10, 10)
        
        # Check if the sample satisfies the condition for class 2
        if 0.5 * X1 - X2 - 10 * X3 + X4 + 50 < 0:
            class2_samples.append([X1, X2, X3, X4])

    # Convert class2_samples to a NumPy array
    class2_samples = np.array(class2_samples)

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

#def case3Samples():
#    case3Class1 = []
#    for i in range(NUMSAMPLES):
#        test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)
#        case3Class1.append(test)
#        # print(test)
#    case3Class2 = []
#    for i in range(NUMSAMPLES):
#        case3Class2.append(np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2))
#    case3Class3 = []
#    for i in range(NUMSAMPLES):
#        case3Class3.append(np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2))
#    case3Class4 = []
#    for i in range(NUMSAMPLES):
#        case3Class4.append(np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2))
#    trainingDataX = []
#    for i in range(NUMSAMPLES):
#        trainingDataX.append(case3Class1[i])
#        trainingDataX.append(case3Class2[i])
#        trainingDataX.append(case3Class3[i])
#        trainingDataX.append(case3Class4[i])
#    # for i in range(NUMSAMPLES):
#    # print(trainingDataX[i])
#    trainingDataY = []
#    for i in range(len(trainingDataX)):
#        if ((0.5 * trainingDataX[i][0]) - trainingDataX[i][1] - (10 * trainingDataX[i][2]) + trainingDataX[i][3] + 50 > 0):
#            trainingDataY.append(1)
#        else:
#            trainingDataY.append(0)
#
#    trainingDataX = np.array(trainingDataX)
#    trainingDataY = np.array(trainingDataY)
#    print("number of instances > 0: ", np.count_nonzero(trainingDataY == 1))
#    print("number of instances < 0: ", np.count_nonzero(trainingDataY == 0))
#
#    return trainingDataX, trainingDataY

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

def makeCurrentPlot(perceptron, xValues, yValues, title, case3=False):
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
    if not case3:
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

    else:
        x_values = np.linspace(MINSAMPLESPACE, MAXSAMPLESPACE, 100)
        w1, w2, w3, w4 = perceptron.weights
        bias = perceptron.bias
        # for each combination of X1, X2, X3 & X4, make a plot like in cases 1 & 2
        for i in range(6):



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

# Case 3

X_train_3, y_train_3 = case3Samples()
perceptron_3 = Perceptron(4,-1,1,THRESHOLD)

perceptron_3.fit(X_train_3, y_train_3, LEARNING_RATE, NUM_EPOCHS)
###makeCurrentPlot(perceptron_3, X_train_3, y_train_3, title='Case 3: Decision Boundary', True)
# Generate test data for Case 2
#X_test_3, y_test_3 = case3Samples()
#misclassified_3 = incorrectlyClassified(perceptron_3, X_test_3, y_test_3)
#print(f"Case 3: Misclassified samples = {misclassified_3}")
