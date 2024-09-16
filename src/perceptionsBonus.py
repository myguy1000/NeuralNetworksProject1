import numpy as np

class Perceptron:
    def __init__(self, input_size, threshold=0):
        # Initialize weights and bias to random numbers in the range of [-1, 1]
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.threshold = threshold

    def forward(self, X_train):
        # Compute the prediction for the class of each input of the perceptron
        curItem=0
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
        # Train the perceptron using the perceptron learning algorithm
        z = range(num_epochs)
        for i in z:
            curInput = 0
            for x in inputs:
                curOutput = output[curInput]
                prediction = self.forward(x)
                self.weights += learning_rate * (curOutput - prediction) * x
                self.bias += learning_rate * (curOutput - prediction)
                curInput += 1

# testing

# iterate through the three test cases
# count: iteration variable; used to switch between test cases
count = 0
for i in range(2): # change to 3 when cases 2 and 3 are complete
    print("beginning of case ", count + 1)

    # 200 samples, two or four features (x1, x2, x3, x4) each with range 1 -> 10
    #samples1_x1 = np.random.uniform(1, 10, 200) 
    samples1_x1 = [None] * 200
    samples1_x2 = [None] * 200
    samples2_x1 = [None] * 200
    samples2_x2 = [None] * 200 #np.random.uniform(1, 10, 200)
    
    for i in range(200):

        #case 1
        if count==0:
            samples1_x1 = np.random.uniform(1, 10, 200)
            samples2_x2 = np.random.uniform(1, 10, 200)

            min_s1x2=samples1_x1[i]
            max_s1x2=10
            min_s2x1=samples2_x2[i]
            max_s2x1=10
            num_classes=2
           

            samples1_x2[i] = np.random.uniform(min_s1x2, max_s1x2, 1) 
            samples2_x1[i] = np.random.uniform(min_s2x1, max_s2x1, 1) 
        
        #case 2
        elif count == 1:
            samples1_x2 = np.random.uniform(2.5, 7.5, 200)
            samples2_x2 = np.random.uniform(2.5, 7.5, 200)
            
            min_s1x1=2*samples1_x2[i]-5
            max_s1x1=7.5
            min_s2x1=2.5
            max_s2x1=2*samples2_x2[i]-5
            num_classes=2
        
            samples1_x1[i] = np.random.uniform(min_s1x1, max_s1x1, 1) 
            samples2_x1[i] = np.random.uniform(min_s2x1, max_s2x1, 1) 

        #else: # count == 2
        #    min1=samples1_x1[i]
        #    max1=2
        #    min2=samples2_x2[i]
        #    max2=
        #    num_classes = 4


    samples1 = np.append(samples1_x1, samples2_x1)
    samples2 = np.append(samples1_x2, samples2_x2)

    y0 = np.zeros(200)
    y1 = y0 + 1
    y_actual = np.append(y1, y0)

    samples_and_y = np.column_stack((samples1, samples2, y_actual))
    np.random.shuffle(samples_and_y)

    # dividing up all samples into train and test
    y_train_actual = samples_and_y[0:200,2]
    y_test_actual = samples_and_y[200:400,2]
    X_train = samples_and_y[0:200,[0,1]]
    X_test = samples_and_y[200:400,[0,1]]
    print("combined x and y: ", samples_and_y)
    
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train_actual shape: ", y_train_actual.shape)
    print("y_test_actual shape: ", y_test_actual.shape)
    #print(X_train)
    #print(y_train)
    
    # testing
    perceptron = Perceptron(100, 2)

    # end of this test, so increment for the next test
    count += 1
    print("")

#perceptron.forward(

# compare predicted with actual y values
#misclassifications = 0
#for i in range(200):
#    if y_actual == perceptron.predictions:
#        misclassifications += 1
#print(misclassifications)


#for i in range(samples_x2):
#
#y_actual = np.zeros(100)
#
## generate the actual y values (class identifier, either 0 or 1) for each sample
#y_actual = np.zeros(100)
#for i in range(100):
#    if -samples[i, 0] + samples[i,1] >= 0:
#        y_actual[i] = 1
#    else:
#        y_actual[i] = 0
#
## generate predictions
#
##case 2
#samples = np.random.uniform(-1, 1, 100) # 100 samples, 2 features (x1, x2) each with range -1 -> 1
#samples = 
#
## generate the actual y values (class identifier, either 0 or 1) for each sample
#y_actual = np.zeros(100)
#for i in range(100):
#    if samples[i, 0] - 2 * samples[i, 1] + 5 >= 0:
#        y_actual[i] = 1
#    else:
#        y_actual[i] = 0
