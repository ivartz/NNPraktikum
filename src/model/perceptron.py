# -*- coding: utf-8 -*-

import sys
import logging
import time

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from report.evaluator import Evaluator
from sklearn.metrics import accuracy_score


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/10

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Here you have to implement the Perceptron Learning Algorithm
        # to change the weights of the Perceptron
        trainingSet = self.trainingSet
        for inputInstance in range(trainingSet.input.shape[1]):
            
            
            # Predefined epoches
            for epoch in xrange(self.epochs):
                error = trainingSet.label[inputInstance] - self.fire(trainingSet.input[inputInstance,:])
                self.weight += self.learningRate*error*trainingSet.input[inputInstance,:]
                #print "Weights adjusted"
            """
            # Error measure or Overfitting detector as epoch termination criteria
            error = trainingSet.label[inputInstance] - self.fire(trainingSet.input[inputInstance,:])
            snapShot = self.weight + self.learningRate*error*trainingSet.input[inputInstance,:]    
            while not np.array_equal(snapShot, self.weight):
                
                error = trainingSet.label[inputInstance] - self.fire(trainingSet.input[inputInstance,:])
                overfitting = accuracy_score(trainingSet.label, self.evaluate(trainingSet.input)) > accuracy_score(self.validationSet.label, self.evaluate(self.validationSet.input))
                
                #accTrain = accuracy_score(trainingSet.label, self.evaluate(trainingSet.input))
                #accValid = accuracy_score(self.validationSet.label, self.evaluate(self.validationSet.input))
                
                #print accTrain

                #print accValid

                if accValid > accTrain:
                    self.weight += self.learningRate*error*trainingSet.input[inputInstance,:]
                    print "Weights adjusted"
            
            # Error measure as epoch termination criteria 
            while True:
                error = trainingSet.label[inputInstance] - self.fire(trainingSet.input[inputInstance,:])
                overfitting = accuracy_score(trainingSet.label, self.evaluate(trainingSet.input)) > accuracy_score(self.validationSet.label, self.evaluate(self.validationSet.input))
                if (error == 0):
                    break
                self.weight += self.learningRate*error*trainingSet.input[inputInstance,:]
                print "Weights adjusted"
            """       
            if verbose:
                evaluator = Evaluator()
                print "<validationSet>",
                evaluator.printAccuracy(self.validationSet, self.evaluate(self.validationSet.input))
                #time.sleep(0.04)




    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Here you have to implement the classification for one instance,
        # i.e., return True if the testInstance is recognized as a 7,
        # False otherwise
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input

        # Here is the map function of python - a functional programming concept
        # It applies the "classify" method to every element of "test"
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return Activation.sign(np.dot(np.array(input), self.weight))
