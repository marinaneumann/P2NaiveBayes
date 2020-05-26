import numpy as np
from sklearn.model_selection import train_test_split
import statistics
#import math
from math import sqrt, pi, exp
import os


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
    #print(os.getcwd())
    data = np.genfromtxt('P2NaiveBayes/spambase/spambase.data', delimiter=',', dtype=float)
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

    #Sets up probablistic model to gather prior probabilities, mean and std. dev.
    def model(self):
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
        global means0, stDevs0, means1, stDevs1
        means0, stDevs0, means1, stDevs1 = arrayStatistics(arrays0, arrays1)

        #Computes prior probability of spam vs not spam
        probSpam = countSpam/ self.numTrain
        probNotSpam = countNot/self.numTrain
        #Output of relevant training set information.
        print("Training data information:")
        print("\t")
        print("Prior probability that it is spam:", probSpam)
        print("Prior probability that it is NOT spam:", probNotSpam)
        print("Standard deviation for class 0:", stDevs0)
        print("Mean for class 0:", means0)
        print("Mean for class 1:", means1)
        print("Standard deviation for class 1:", stDevs1)
        print("\t")

    #Function for Gaussian Naive bayes Algorithm
    def naiveBayesAlg(self):
        print("Test Data")
        countSpam = 0
        countNot = 0
        for i in self.test:
            if i[57] == 1:
                countSpam +=1
            else:
                countNot +=1
        probNum0 = countNot / self.numTest
        probNum1 = countSpam / self.numTest
        accuracy = 0
        acc = 0
        prediction = 99
        for i in self.test:
            p0 = np.log(probNum0) + calculateProb(i,means0, stDevs0)
            #p0 = np.argmax(p0)
            p1 = np.log(probNum1)+ calculateProb(i, means1, stDevs1)
            #p1 = np.argmax(p1)
            test = max(p0, p1)
            if(test == p0):
                prediction = 0
            elif(test == p1):
                prediction = 1
            acc = accuracyTest(i, prediction)
            print("Data SHOULD be", i[57])
            if(res == True):
                print("Correctly predicted")
            else:
                print("Incorrectly predicted")
            print("\t")
            accuracy +=acc

        print("\t")
        print("Accuracy is:", (accuracy/self.numTest)*100)  #True positive & true negative divided by all
        print("Precision is:")
        print("Total recall is:")

def accuracyTest(data,prediction):
   global res


   if data[57] == prediction:
       res = True #Generally reports to user if correctly predicted or not
       return 1
   else:
       res = False
       return 0

def calculateProb(data, me, sD):
    # pi = math.pi
    product =1
    for j in range(0,57):
        ex = exp(-((data[j]- me[j])**2/(2*sD[j]**2)))
        p =(1/ (sqrt(2* pi)* sD[j]))* ex
        #p = (1/sqrt(2*pi*sD[j]))* exp(-0.5*pow((data[j]-me[j]),2)/sD[j])
        if p == 0:
            p = 0.000000000001
        #product = product + np.log(p)
        product = product + np.log(p)
    return product

def arrayStatistics(arrays0, arrays1):
    means0 = []
    means1 =[]
    stDevs0 =[]
    stDevs1 =[]
    for column in zip(*arrays0):
        means0.append(mean(column))
        stDevs0.append(stdev(column))
    del (means0[-1])
    del (stDevs0[-1])

    for column in zip(*arrays1):
        means1.append(mean(column))
        stDevs1.append(stdev(column))
    del (means1[-1])
    del (stDevs1[-1])
    return means0, stDevs0, means1, stDevs1

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