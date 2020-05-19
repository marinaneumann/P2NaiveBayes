import numpy as np
from sklearn.model_selection import train_test_split
import statistics
#import math
from math import sqrt, pi, exp


def main():
    print("Programming Assignment #2 for CS445")
    print("By Marina Neumann ")
    print("Spring 2020")
    dataLoad()  #function to load data from spambase
    spamOrNot = spamClassifer(Xtrain,Xtest)  #establishment of spamClassifer object

    spamOrNot.model() #creates probabilistic model
    spamOrNot.naiveBayesAlg()

#Loads function from data and splits into training and test data with a 50/50 split so roughly 2300 data instances per set
def dataLoad():
    global Xtrain, Xtest
    data = np.genfromtxt('spambase/spambase.data', delimiter=',', dtype=float)
    Xtrain, Xtest =train_test_split(data, shuffle = True, test_size = 0.5)

#SpamClassifer object established
class spamClassifer():
    #Initialized data inputs and sizes to be used later in computation
    def __init__(self,train,test):
        self.train = train
        self.test = test
        self.size = 58
        self.numTrain = len(train)
        self.numTest = len(test)
        #Below is used solely for the means of using the algorithm on test data
        # arrays1 = []
        # arrays0 = []
        # for i in self.test:
        #     if i[57] == 1:
        #         arrays1.append(i)
        #     else:
        #         arrays0.append(i)
        # self.test0 = arrays0
        # self.test1 = arrays1
        # print( "WTF?",arrays0[0][57])
    #Sets up probablistic model to gather prior probabilities, mean and std. dev.
    def model(self):
        #Variables below used to distinguish between separate means for whether 0 or 1 class.
        means0 = []
        stDevs0 = []
        means1 = []
        stDevs1 = []
        #Used for splitting of self.train into arrays of class 0 and arrays of class 1
        arrays0 =[]
        arrays1 =[]
        #variables to track count of labels as 0 or 1 to find prior probabilities
        countSpam = 0
        countNot = 0
        #Splits data based on classification of 0 or 1, in hopes of simplifying mean/std dev process
        for i in self.train:
            if i[57] == 1:
                arrays1.append(i)
                countSpam +=1
            else:
                arrays0.append(i)
                countNot +=1
        #Used to take mean/std deviation of all arrays with class 0
        for column in zip(*arrays0):
            means0.append(mean(column))
            stDevs0.append(stdev(column))
        #Below deletes means and std deviations of 58th element as that takes mean/std dev of class labels (0 or 1)
        del(means0[-1])
        del(stDevs0[-1])
        # Used to take mean/std deviation of all arrays with class 1
        for column in zip(*arrays1):
            means1.append(mean(column))
            stDevs1.append(stdev(column))
        #Deletes mean/std dev of clas labels as unnecessary
        del (means1[-1])
        del (stDevs1[-1])
        #Computes prior probability of spam vs not spam
        probSpam = countSpam/ self.numTrain
        probNotSpam = countNot/self.numTrain
        #Output of relevant training set information.
        print("Prior probability that it is spam:", probSpam)
        print("Prior probability that it is NOT spam:", probNotSpam)
        print("Standard deviation for class 0:", stDevs0)
        print("Mean for class 0:", means0)
        print("Mean for class 1:", means1)
        print("Standard deviation for class 1:", stDevs1)

    #Function for Gaussian Naive bayes Algorithm
    def naiveBayesAlg(self):
        print("SUP?")
        countSpam = 0
        countNot = 0
        arrays0  =[]
        arrays1 =[]
        for i in self.test:
            if i[57] == 1:
                arrays1.append(i)
                countSpam +=1
            else:
                arrays0.append(i)
                countNot +=1
        means0 = []
        stDevs0 = []
        means1 = []
        stDevs1 = []

        probabilities = []
        probabilities0 = []
        probabilities1 = []
        finalprob0 = [] #Will hold all class 0(x) probabilities
        finalprob1 = [] #Will hold all class 1(x) probabilities
        for column in zip(*arrays0):
            means0.append(mean(column))
            stDevs0.append(stdev(column))
        del (means0[-1])
        del (stDevs0[-1])
        cNum = 1 #cNums are used as indicators for testClassProb function, to specify what is to be returned later

        for i in arrays0:
            for z in range(0,57):
                probs = calculateProb(i[z], means0[z], stDevs0[z])
                probabilities.append(probs)
            probabilities0.append(np.log(probabilities))
        print(probabilities0)
        probNum0 = countNot / self.numTest
        #finalprob0.append(np.argmax(sum(probabilities0, np.log(probNum0))))
        # #print("Final probabilities of class 0:", finalprob0)

        probabilities.clear()
        for column in zip(*arrays1):
            means1.append(mean(column))
            stDevs1.append(stdev(column))
        del (means1[-1])
        del (stDevs1[-1])
        cNum =0
        for m in arrays1:
            for z in range(0,57):
                probs = calculateProb(m[z], means1[z], stDevs1[z])
                #probabilities.append(probs)
            probabilities1.append(np.log(probabilities))
        print(probabilities1)
        probNum1 = countSpam/self.numTest

        #finalprob1.append(np.argmax(sum(probabiltiies1, probNum1)))
        #print("Final probabilities of class1:", finalprob1)

#Function used to calculate probability of X given class (0 or 1).
#Used Xi, then mean and st. deviation of all x's with that classification
def calculateProb(data, me, sD):
    # pi = math.pi
    ex = exp(-((data- me)**2/(2*sD**2)))
    return (1/ (sqrt(2* pi)* sD))* ex

#Calculates probability of class for test set
# def testClassProb (data,cNum ):
#     countSpam = 0
#     countNot = 0
#     for i in data:
#         if i[57] == 1:
#             countSpam +=1
#         else:
#             countNot +=1
#     print(countSpam)
#     if cNum == 1:
#         print(countSpam/len(data))
#         return countSpam/len(data)
#     else:
#         return countNot/len(data)

#Function to calculate the mean of data passed in
def mean(data):
    return sum(data)/float(len(data))

#Function to calculate the st. deviation for data passed in
def stdev(data):
    avg = mean(data)
    variance = sum([(x-avg)**2 for x in data])/ float(len(data)-1)
    num = sqrt(variance)
    if num == 0: #Used to avoid zero standard deviation by assigning minimal SD, to avoid possible zero error later.
        return 0.0001
    else:
        return num

#Calls main/executes program
main()