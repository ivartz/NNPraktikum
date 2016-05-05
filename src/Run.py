#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
#from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    #myStupidClassifier = StupidRecognizer(data.trainingSet,
#                                          data.validationSet,
#                                          data.testSet)
    # Uncomment this to make your Perceptron evaluated
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                         data.validationSet,
                                         data.testSet,
                                         learningRate=0.8,
                                         epochs=4)

    # Train the classifiers
    print("=========================")
    print("Training..")

    #print("\nStupid Classifier has started training..")
    #myStupidClassifier.train()
    #print("Done..")

    print("\nPerceptron has started training..")
    myPerceptronClassifier.train(True)
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    #stupidPred = myStupidClassifier.evaluate()
    # Uncomment this to make your Perceptron evaluated
    perceptronPredOnTrainingSet = myPerceptronClassifier.evaluate(data.trainingSet.input)
    perceptronPredOnValidationSet = myPerceptronClassifier.evaluate(data.validationSet.input)
    perceptronPredOnTestSet = myPerceptronClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    #print("Result of the stupid recognizer:")
    #evaluator.printComparison(data.testSet, stupidPred)
    #evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    #evaluator.printComparison(data.testSet, perceptronPred)
    # Uncomment this to make your Perceptron evaluated
    
    print "<trainingSet>",
    evaluator.printAccuracy(data.trainingSet, perceptronPredOnTrainingSet)

    print "<validationSet>",
    evaluator.printAccuracy(data.validationSet, perceptronPredOnValidationSet)

    print "<testSet>",
    evaluator.printAccuracy(data.testSet, perceptronPredOnTestSet)

    # eval.printConfusionMatrix(data.testSet, pred)
    # eval.printClassificationResult(data.testSet, pred, target_names)

if __name__ == '__main__':
    main()
