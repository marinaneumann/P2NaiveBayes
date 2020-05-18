import numpy as np


def main():
    print("Programming Assignment #2 for CS445")
    print("By Marina Neumann ")
    print("Spring 2020")
    dataLoad()

def dataLoad():
    data = np.genfromtxt('spambase/spambase.data')
    #Xtrain, Xtest, Ytrain, Ytest = train_test_split(data)
    print(data)


def model():
    print("blehhhh")
main()