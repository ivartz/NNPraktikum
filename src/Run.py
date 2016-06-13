#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from data.mnist_seven import MNISTSeven

from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot

def main():
    # load data
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                       one_hot=False)

    # Train the classifiers #
    print("=========================")
    print("Training..")

    # Stupid Classifier
    #myStupidClassifier = StupidRecognizer(data.training_set,
    #                                      data.validation_set,
    #                                      data.test_set)

    #print("\nStupid Classifier has been training..")
    #myStupidClassifier.train()
    #print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    #stupidPred = myStupidClassifier.evaluate()

    # Perceptron
    #myPerceptronClassifier = Perceptron(data.training_set,
    #                                    data.validation_set,
    #                                    data.test_set,
    #                                    learning_rate=0.005,
    #                                    epochs=10)

    #print("\nPerceptron has been training..")
    #myPerceptronClassifier.train()
    #print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    #perceptronPred = myPerceptronClassifier.evaluate()

    # Logistic Regression
    #myLRClassifier = LogisticRegression(data.training_set,
    #                                    data.validation_set,
    #                                    data.test_set,
    #                                    learning_rate=0.005,
    #                                    epochs=30)

    #print("\nLogistic Regression has been training..")
    #myLRClassifier.train()
    #print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    #lrPred = myLRClassifier.evaluate()

    # two layer mlp parameters
    neuronsInHiddenLayer = 64
    epochs =  100
    learningRate =  0.01

    if (len(sys.argv) == 4):
        neuronsInHiddenLayer = int(sys.argv[1])
        epochs =  int(sys.argv[2])
        learningRate =  float(sys.argv[3])

    myMLPClassifier = MultilayerPerceptron(data.training_set,
                                           data.validation_set,
                                           data.test_set,
                                           learning_rate=learningRate,
                                           epochs=epochs,
                                           hidden_neurons=neuronsInHiddenLayer)


    print("\nLogistic Regression has been training..")
    myMLPClassifier.train()
    print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()

    # Report the result #
    print("=========================")
    evaluator = Evaluator()

    #print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    #evaluator.printAccuracy(data.test_set, stupidPred)

    #print("\nResult of the Perceptron recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    #evaluator.printAccuracy(data.test_set, perceptronPred)

    #print("\nResult of the Logistic Regression recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    #evaluator.printAccuracy(data.test_set, lrPred)

    print("\nResult of the Multi-layer Perceptron recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.test_set, mlpPred)

    # Draw
    plot = PerformancePlot("MLP validation",
                            myMLPClassifier.performances,
                            myMLPClassifier.epochs,
                            myMLPClassifier.learning_rate,
                            myMLPClassifier.hidden_neurons,
                            evaluator.getAccuracy(data.test_set, mlpPred))

    plot.save_performance_as_img(folder="./plots")
    plot.draw_performance_epoch()

if __name__ == '__main__':
    main()
