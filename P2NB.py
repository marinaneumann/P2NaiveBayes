import numpy as np
from sklearn.model_selection import train_test_split
import statistics
#import math
from math import sqrt, pi, exp


def main():
    print("Programming Assignment #2 for CS445")
    print("By Marina Neumann ")
    print("Spring 2020")
    dataLoad()
    spamOrNot = spamClassifer(Xtrain,Xtest)

    spamOrNot.model()
    spamOrNot.naiveBayesAlg()

def dataLoad():
    global Xtrain, Xtest
    data = np.genfromtxt('spambase/spambase.data', delimiter=',', dtype=float)
    Xtrain, Xtest =train_test_split(data, shuffle = True, test_size = 0.5)


class spamClassifer():
    def __init__(self,train,test):
        self.train = train
        self.test = test
        self.size = 58
        self.numTrain = len(train)
        self.numTest = len(test)

        arrays1 = []
        arrays0 = []
        for i in self.test:
            if i[57] == 1:
                arrays1.append(i)
            else:
                arrays0.append(i)
        self.test0 = arrays0
        self.test1 = arrays1

    def model(self):
        means0 = []
        stDevs0 = []
        means1 = []
        stDevs1 = []

        arrays0 =[]
        arrays1 =[]
        countSpam = 0
        countNot = 0
        for i in self.train:
            if i[57] == 1:
                arrays1.append(i)
                countSpam +=1
            else:
                arrays0.append(i)
                countNot +=1
        for column in zip(*arrays0):
            means0.append(mean(column))
            stDevs0.append(stdev(column))
        del(means0[-1])
        del(stDevs0[-1])
        for column in zip(*arrays1):
            means1.append(mean(column))
            stDevs1.append(stdev(column))
        del (means1[-1])
        del (stDevs1[-1])
        probSpam = countSpam/ self.numTrain
        probNotSpam = countNot/self.numTrain
        print("Prior probability that it is spam:", probSpam)
        print("Prior probability that it is NOT spam:", probNotSpam)
        print("Standard deviation for class 0:", stDevs0)
        print("Mean for class 0:", means0)
        print("Mean for class 1:", means1)
        print("Standard deviation for class 1:", stDevs1)


    def naiveBayesAlg(self):
        print("SUP?")
        #means = []
        #stDevs = []
        probabilities0 = []
        probabiltiies1 = []
        finalprob0 =0
        finalprob1 = 0
        for i in self.test0:
            me =mean(i)
            sD =stdev(i)
            probs = calculateProb(i, me, sD)
            probabilities0.append(probs)
        cNum =1
        probNum1 = testClassProb(self.test0, cNum)
        finalprob0 =  np.argmax(sum(probabilities0, probNum1))
        print("Final probability of class 0:", finalprob0)
        for m in self.test1:
            me = mean(i)
            sD = stdev(i)
            probs = calculateProb(i, me, sD)
            probabiltiies1.append(probs)
        cNum = 0
        probNum1 = testClassProb(self.test1, cNum)
        finalprob1 = np.argmax(sum(probabiltiies1, probNum1))
        print("Final probabilitie of class1:", finalprob1)

def calculateProb(data, me, sD):
    # pi = math.pi
    ex = exp(-((data- me)**2/(2*sD**2)))
    return (1/ (sqrt(2* pi)* sD))* ex

def testClassProb (data,cNum ):
    countSpam = 0
    countNot = 0
    for i in data:
        if i[57] == 1:
            countSpam +=1
        else:
            countNot +=1
    if cNum == 1:
        return countSpam/len(data)
    else:
        return countNot/len(data)

def mean(data):
    return sum(data)/float(len(data))

def stdev(data):
    avg = mean(data)
    variance = sum([(x-avg)**2 for x in data])/ float(len(data)-1)
    return sqrt(variance)

main()