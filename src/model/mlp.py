
import numpy as np

from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 cost='crossentropy', learning_rate=0.01, epochs=50):

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
        self.input_weights = input_weights

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        input_activation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 100, None, input_activation, False)) 

        # Hidden layer
        #hidden_activation = "sigmoid"
        #self.layers.append(LogisticLayer(200, 100, None, hidden_activation, False)) 

        # Output layer
        output_activation = "softmax"
        self.layers.append(LogisticLayer(100, train.output.shape[1], None, output_activation, True))

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self.get_layer(0)

    def _get_output_layer(self):
        return self.get_layer(-1)

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
        # list to record activations, appending numpy array results to it
        activationValuesInLayers = []
        #for layer_index in range(self.layers):
        #    if layer_index != 0:
        #        inp = activationValuesInLayers[layer_index-1]
        #    activationValues = self._get_layer(layer_index).forward(inp)
        #    activationValuesInLayers.append(activationValues)
        #return activationValuesInLayers[-1]

        for layer_index, layer in enumerate(self.layers):
            if layer_index != 0:
                inp = activationValuesInLayers[layer_index-1]
            activationValues = layer.forward(inp)
            activationValuesInLayers.append(activationValues)

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        #self.outputLayerError = BinaryCrossEntropyError.calculate_error(target - self._get_output_layer().outp)
        #return self.outputLayerErrornext_weights

        # computing error terms for the latest layer
        self._get_output_layer().computeErrorTerms( target - self._get_output_layer().outp, np.array(1.0) )

    def _update_weights(self):
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




        """Train the Logistic Regression.

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

        pass



        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.training_set.input,
                              self.training_set.label):

            # Use LogisticLayer to do the job
            # Feed it with inputs

            # Do a forward pass to calculate the output and the error
            self.layer.forward(img)

            # Compute the derivatives w.r.t to the error
            # Please note the treatment of nextDerivatives and nextWeights
            # in case of an output layer
            self.layer.computeErrorTerms(np.array(label - self.layer.outp),
                                         np.array(1.0))

            # Update weights in the online learning fashion
            self.layer.updateWeights(self.learning_rate)



    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        return True

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
