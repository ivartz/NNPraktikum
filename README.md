### NNPraktikum Übung 1
A simple implementation of the Rosenblatt perceptron with Heaviside step function to recognize a handwritten digit
whether it is a seven (7) or not, using the picture samples from the MNIST dataset. An additional stupid recognizer using random classification is also implemented for comparision.

Remarks on learningRate and epocs Parameters: There is a tradeoff between the learningRate and epocs parameters regarding to the classifying accuracy. An additioal factor is training execution speed. It appears that larger learningRate leads to higher oscillations in evaluated accuracy using the validation set during training. Hence it can lead to more unpredictable classifying accuracy on the test set. Achieving similar accuracy on the test- and validation sets by lowering the learninRate requires more epoches to be run in the perception training for each input instance. This also leads to longer training execution time. Some parameters for reasonable accuracy and fast training execution time are learningRate=0.8 , epochs=4.

#### KIT Neural Network Framework

See [ilias.studium.kit.edu](https://ilias.studium.kit.edu/goto_produktiv_crs_413999.html)
for more information.

Requirements: Python 2.7.x, Numpy, scikit-learn