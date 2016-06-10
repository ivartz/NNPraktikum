
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

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        input_activation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 64, None, input_activation, False)) # train.input.shape[1]-1 because input data includes 

        # Hidden layer
        #hidden_activation = "sigmoid"
        #self.layers.append(LogisticLayer(200, 100, None, hidden_activation, False)) 

        # Output layer
        output_activation = "softmax"
        self.layers.append(LogisticLayer(64, 10, None, output_activation, True)) # 10 if not len(set(train.input)) working len(set(train.input)

        #print train.input.shape       

        #sys.exit()

        self.input_weights = input_weights

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)




        #print self.layers[0].shape
        #print self.layers[1].shape

        

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
        # list to record activations, appending numpy array results to it
        #activationValuesInLayers = []
        #for layer_index in range(self.layers):
        #    if layer_index != 0:
        #        inp = activationValuesInLayers[layer_index-1]
        #    activationValues = self._get_layer(layer_index).forward(inp)
        #    activationValuesInLayers.append(activationValues)
        #return activationValuesInLayers[-1]

        #for layer_index, layer in enumerate(self.layers):
        #    if layer_index != 0:
        #        inp = activationValuesInLayers[layer_index-1]
        #    activationValues = layer.forward(inp)
        #    activationValuesInLayers.append(activationValues)

        # another method that stores a layers output within itself
        for layer_index, layer in enumerate(self.layers):
            #print "working on layer ", layer_index

            if layer_index != 0:
                inp = self._get_layer(layer_index-1).outp
                inp =  np.insert(inp, 0, 1, axis=0)
                #print "set inp to the previous layers output"
            #print inp.shape
            #print layer.shape
            layer.forward(inp)
            
            #outp = layer.forward(inp)
            #outp.insert(outp, 0, 1, axis = 0)


        # Dominik





    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        #self.outputLayerError = BinaryCrossEntropyError.calculate_error(target - self._get_output_layer().outp)
        #return self.outputLayerErrornext_weights

        # computing error terms for the latest layer
        #self._get_output_layer().computeErrorTerms( target - self._get_output_layer().outp, np.array(1.0) )

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
            #print "___HERE______HERE______HERE______HERE______HERE______HERE______HERE______HERE______HERE___"



            #newInput = inp
            # for all layers
            #for l in self.layers:
            #    newInput = l.forward(newInput)
            #    # add bias values ("1"s) at the beginning of all data sets
            #    newInput = np.insert(newInput, 0, 1, axis=0)
            #    
            #return newInput


            #print img.shape

            #sys.exit()
            self._feed_forward(img)

            #print "___HERE2___HERE2___HERE2___HERE2___HERE2___HERE2___HERE2___HERE2___HERE2___HERE2___HERE2"


            # compute error terms of the output layer
            labelList = [0]*10 # todo: 10 not hardcoded
            labelList[label] = 1
            
            #print labelList
            #print label

            self._compute_error(labelList)


            # backwards iteratively compute the error terms in the other (hidden) layers w.r.t to the error terms in the next layer
            for layer_index, layer in reversed(list(enumerate(self.layers))):
                if (not layer.is_classifier_layer) and layer_index != 2:   # todo: 2 not hardcoded
                    
                    next_layer = self._get_layer(layer_index+1)
                    next_layer_error_terms = next_layer.errorTerms
                    next_layer_weights = next_layer.weights
                    
                    #print "___HERE______HERE______HERE______HERE______HERE______HERE______HERE______HERE______HERE___"

                    #print next_layer_weights.shape
                    #print next_layer_weights[1::].shape
                    
                    #sys.exit()
                    layer.computeErrorTerms(next_layer_error_terms, np.transpose(next_layer_weights[1::]))
            
            # Update weights in the online learning fashion
            self._update_weights(self.learning_rate)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        self._feed_forward(test_instance)
        
        outp = self._get_output_layer().outp
        
        #print outp

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
