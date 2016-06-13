import matplotlib.pyplot as plt

class PerformancePlot(object):
    '''
    Class to plot the performances
    Very simple and badly-styled
    '''
    def __init__(self, name, performances, epochs, learning_rate,
                 hidden_neurons, testAccuracy):

        '''
        Constructor
        '''
        self.name = name

        self.performances = performances
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_neurons = hidden_neurons
        self.testAccuracy = testAccuracy

        plt.plot(range(self.epochs), self.performances, 'k',
                 range(self.epochs), self.performances, 'ro')

        plt.title("Performance of " + self.name + " over the epochs")
        plt.ylim(ymax=1)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")

    def draw_performance_epoch(self):
        plt.show()

    def save_performance_as_img(self, folder):
        plt.savefig(folder+"/"+
                    str(self.epochs) + "Epochs_"+
                    str(self.hidden_neurons)+"Neurons_"+
                    str(self.learning_rate)+"LearningRate"+
                    str(self.testAccuracy)+"TestAccuracy.png")

