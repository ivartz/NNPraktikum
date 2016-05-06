# -*- coding: utf-8 -*-

from random import random
from model.classifier import Classifier

__author__ = "ugeuh"  # Adjust this when you copy the file
__email__ = "ugeuh@student.kit.edu"  # Adjust this when you copy the file


class StupidRecognizer(Classifier):
    """
    This class demonstrates how to follow an OOP approach to build a digit
    recognizer.

    It also serves as a baseline to compare with other
    recognizing method later on.

    The method is that it will randomly decide the digit is a "7" or not
    based on the probability 'byChance'.
    """

    def __init__(self, train, valid, test, byChance=0.5):

        self.byChance = byChance

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

    def train(self):
        # Do nothing
        pass

    def classify(self, testInstance):
        # byChance is the probability of being correctly recognized
        return random() < self.byChance

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
        
        return list(map(self.classify, test))
