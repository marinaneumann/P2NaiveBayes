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
    #spamOrNot.naiveBayesAlg()

def dataLoad():
    global Xtrain, Xtest
    data = np.genfromtxt('spambase/spambase.data', delimiter=',', dtype=float)
    Xtrain, Xtest =train_test_split(data, shuffle = True, test_size = 0.5)

    #print(Xtrain.shape)
    #print(Xtest.shape)

class spamClassifer():
    def __init__(self,train,test):
        self.train = train
        self.test = test
        self.size = 58
        self.numTrain = len(train)
        self.numTest = len(test)


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

        # for i in self.train:
        #     if i[57] ==1:
        #         arrays1.append(i)
        #         countSpam += 1
        #     else:
        #         arrays0.append(i)
        #         countNot += 1
        # z =0
        # x = 0
        # #for i in arrays0:
        # for z in range(self.size -1):
        #     sD0 = np.std(arrays0, axis = 0)
        #     stDevs0.append(sD0)
        # for x in range(self.size -1):
        #     sD1 = np.std(arrays1, axis = 0)
        #     stDevs1.append(sD1)
        #
        probSpam = countSpam/ self.numTrain
        probNotSpam = countNot/self.numTrain
        print("Prior probability that it is spam:", probSpam)
        print("Prior probability that it is NOT spam:", probNotSpam)
        # print("Standard deviation for class 0:", stDevs0)
        # print("Standard deviation for class 0 length:", len(stDevs0))

        # for i in range(self.size -1):
        #     sD = np.std(Xtrain, axis = 0)
        #     # if sD == 0:
        #     #     sD = 0.0001
        #     #     stDevs0.append(sD)
        #     # else:
        #     #     stDevs0.append(sD)
        #     stDevs.append(sD)
        # print("Standard deviations for each?:", stDevs)
        # print("Number of standard devs:", len(stDevs))


    def naiveBayesAlg(self):
        print("SUP?")
        #me =mean(self.test)
        #sD =stdev(self.test)
        #calculateProb(self, me, sD)

    def calculateProb(data, me, sD):
        pi = math.pi
        ex = exp(-((data- me)**2/(2*sD**2)))
        return (1/ (sqrt(2* pi)* sD))* ex

def mean(data):
    return sum(data)/float(len(data))

def stdev(data):
    avg = mean(data)
    variance = sum([(x-avg)**2 for x in data])/ float(len(data)-1)
    return sqrt(variance)

main()