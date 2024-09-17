import numpy as np
import matplotlib.pyplot as plt
from perceptions import Perceptron  # Ensure the class is saved as current_perceptron.py

THRESHOLD = 0
#the magnitude at which the models weights are updated during a single increment
#lower learning rate may lead to a more accurate result but longer to reach that result
LEARNING_RATE = 0.1
#how many times the dataset will be passed through the model
NUM_EPOCHS = 50
NUMSAMPLES = 100
#min value a sample can take
MINSAMPLESPACE = -100
#max value a sample can take
MAXSAMPLESPACE = 100


"""
case1Samples generates the sample data to be used by the perception class
Generates  100 samples in Class 1: {(x1, x2) | −x1 + x2 > 0} 
and 100 samples in Class 2: {(x1, x2) | − x1 + x2 < 0}
Returns two numpy arrays, for x and y
"""
def case1Samples():
    case1Class1 = []
    for i in range(NUMSAMPLES):
        correct = 0
        while correct == 0:
            test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)
            if (-1 * test[0]) + test[1] > 0:
                case1Class1.append(test)
                correct = 1
            else:
                correct = 0
        # print(test)
    case1Class2 = []
    for i in range(NUMSAMPLES):
        correct = 0
        while correct == 0:
            test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)

            if (-1 * test[0]) + test[1] < 0:
                case1Class2.append(test)
                correct = 1

            else:
                correct = 0
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


"""
case2Samples generates the sample data to be used by the perception class
Generates 100 samples in Class 1: {(x1, x2) | x1 −
2x2 + 5 > 0} and 100 samples in Class 2: {(x1, x2) | x1 − 2x2 + 5 < 0}
Returns two numpy arrays, for x and y
"""
def case2Samples():
    case2Class1 = []
    for i in range(NUMSAMPLES):
        correct = 0
        while correct == 0:
            test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)

            if test[0] - (2 * test[1]) + 5 > 0:
                case2Class1.append(test)
                correct = 1
            else:
                correct = 0
        # print(test)
    case2Class2 = []
    for i in range(NUMSAMPLES):
        correct = 0
        while correct == 0:
            test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)
            if test[0] - (2 * test[1]) + 5 < 0:
                case2Class2.append(test)
                correct = 1
            else:
                correct = 0
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

"""
case3Samples generates the sample data to be used by the perception class
Generates 100 samples in Class 1: {(x1, x2, x3, x4) | 0.5x1 − x2 − 10x3 + x4 + 50 > 0}
and 100 samples in Class 2: {(x1, x2, x3, x4) | 0.5x1 − x2 − 10x3 + x4 + 50 < 0}
Returns two numpy arrays, for x and y
"""
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
    X_shuffled = combined[:, :-1] # All columns except the laast one are X
    y_shuffled = combined[:, -1] # The last column is y

    return X_shuffled, y_shuffled

"""
incorrectlyClassified examines each data point and determines if it is correctly classified into the correct class
returns an integer number that is the number of data points/samples that have been misclassified
"""
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


"""
Uses matplot lib to make a plot that shows the data points (colored) and the separation hyperplane
plots in 2d, examining x1 vs x2 in our case.
Key provided to show which class the data is in
"""
def makeCurrentPlot(current_perceptron, xValues, yValues, title):
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

    plt.scatter(graphX, graphY, color='black', label='Class 1')
    plt.scatter(graphX2, graphY2, color='green', label='Class 2')

    #samplevalues for number of x data points, should be the same as number of y values
    samplevalues = 100
    x_values = np.linspace(MINSAMPLESPACE, MAXSAMPLESPACE, samplevalues)
    w1 = current_perceptron.weights[0]
    w2 = current_perceptron.weights[1]
    #grab the bias value from the perception class
    bias = current_perceptron.getBias()
    part1 = -1 * (w1/w2)
    part2 = part1 * x_values
    biasDivWeight2 = bias/w2
    y_values = part2 - biasDivWeight2
    plt.plot(x_values, y_values, 'b--', label='Our Separation Hyperplane')

    #Labeling our plot and setting the tile
    plt.title(title)
    plt.xlabel('X1 values')
    plt.ylabel('X2 values')
    plt.legend()


    plt.show()

#Generate data for case #1
trainingDataX1, trainingDataY1 = case1Samples()
input_size_1 = 2
maxP = 1
minP = -1
#generate Perceptron with input values
perceptron_1 = Perceptron(input_size_1,minP,maxP,THRESHOLD)
#Train perceptron using fit
perceptron_1.fit(trainingDataX1, trainingDataY1, LEARNING_RATE, NUM_EPOCHS)

#generate sample data
X_test_1, Y_test_1 = case1Samples()
#Count the number of misclassified data samples and print
misclassified_1 = incorrectlyClassified(perceptron_1, X_test_1, Y_test_1)
print("The number of samples that have been misclassified for case #1 is: " + str(misclassified_1))

#Generate the plot for our Case #1
makeCurrentPlot(perceptron_1, X_test_1, Y_test_1, title='Case 1: Decision Boundary')


#Generate Data for Case#2
trainingDataX2, trainingDataY2 = case2Samples()
input_size_2 = 2
maxP = 1
minP = -1
#create perceptron for case 2
perceptron_2 = Perceptron(input_size_2,minP,maxP,THRESHOLD)

#train perceptron2 from the sample cases
perceptron_2.fit(trainingDataX2, trainingDataY2, LEARNING_RATE, NUM_EPOCHS)
#plot the data and make test samples
X_test_2, Y_test_2 = case2Samples()
misclassified_2 = incorrectlyClassified(perceptron_2, X_test_2, Y_test_2)
print("The number of samples that have been misclassified for case #2 is: " + str(misclassified_2))
makeCurrentPlot(perceptron_2, X_test_2, Y_test_2, title='Case 2: Decision Boundary')

# Case 3

X_train_3, y_train_3 = case3Samples()
perceptron_3 = Perceptron(4,-1,1,THRESHOLD)

# regular fit function


#train
perceptron_3.fit(X_train_3, y_train_3, LEARNING_RATE, NUM_EPOCHS)
misclassified_3 = incorrectlyClassified(perceptron_3, X_train_3, y_train_3)
print(f"Case 3a: Misclassified samples reg-fit (train) = {misclassified_3}")

#test
X_test_3, y_test_3 = case3Samples()

# Generate test data for Case 2
misclassified_3 = incorrectlyClassified(perceptron_3, X_test_3, y_test_3)
print(f"Case 3b: Misclassified samples reg-fit (test) = {misclassified_3}")


# GD fit function


#train
perceptron_3.fit_GD(X_train_3, y_train_3, LEARNING_RATE, NUM_EPOCHS)
misclassified_3 = incorrectlyClassified(perceptron_3, X_train_3, y_train_3)
print(f"Case 3c: Misclassified samples GD-fit (train) = {misclassified_3}")

#test
# Generate test data for Case 2
misclassified_3 = incorrectlyClassified(perceptron_3, X_test_3, y_test_3)
print(f"Case 3d: Misclassified samples GD-fit (test) = {misclassified_3}")
