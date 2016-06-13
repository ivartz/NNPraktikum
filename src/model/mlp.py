
import numpy as np

from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """
    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 cost='crossentropy', learning_rate=0.01, epochs=50,
                 hidden_neurons=64):
        """
        A digit-7 recognizer based on logistic regression algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.hidden_neurons = hidden_neurons

        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        input_activation = "sigmoid"

        self.layers.append(LogisticLayer(train.input.shape[1], self.hidden_neurons, None, input_activation, False))

        # Hidden layer
        #hidden_activation = "sigmoid"
        #self.layers.append(LogisticLayer(200, 100, None, hidden_activation, False))

        # Output layer
        output_activation = "softmax"
        self.layers.append(LogisticLayer(self.hidden_neurons, 10, None, output_activation, True))


        self.input_weights = input_weights

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        for layer_index, layer in enumerate(self.layers):
            #print "working on layer ", layer_index
            if layer_index != 0:
                inputExludingBias = self._get_layer(layer_index-1).outp
                inp =  np.insert(inputExludingBias, 0, 1, axis=0)
                #print "set inp to the previous layers output"
            layer.forward(inp)

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Sets
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        # computing error terms for the latest layer
        self._get_output_layer().errorTerms = target - self._get_output_layer().outp

    def _update_weights(self, learning_rate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(self.learning_rate)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        for img, label in zip(self.training_set.input,
                              self.training_set.label):
            # do a feed-forward to calculate the output and the error of the entire network

            self._feed_forward(img)

            # compute error terms of the output layer
            labelList = [0]*10 # todo: 10 not hardcoded
            labelList[label] = 1

            self._compute_error(labelList)

            # backwards iteratively compute the error terms in the other (hidden) layers w.r.t to the error terms in the next layer
            for layer_index, layer in reversed(list(enumerate(self.layers))):
                if (not layer.is_classifier_layer):# and layer_index != 2:   # todo: 2 not hardcoded


                    next_layer = self._get_layer(layer_index+1)
                    next_layer_error_terms = next_layer.errorTerms
                    next_layer_weights = next_layer.weights

                    layer.computeErrorTerms(next_layer_error_terms, np.transpose(next_layer_weights[1::]))

            # Update weights in the online learning fashion
            self._update_weights(self.learning_rate)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        self._feed_forward(test_instance)

        outp = self._get_output_layer().outp

        return np.argmax(outp)

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
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
