"""
File: trainMLP.py
Language: Python 3.5.1
Author: Aravindh Kuppusamy ( axk8776@rit.edu )
        Karan Jariwala( kkj1811@rit.edu )
        Deepak Sharma ( ds5930@rit.edu )
Description: It takes a file containing training data as input and trained
             neural network weights after some epochs. It uses batch
             gradient descent to trained neural network weights.
"""

__author__ = "Aravindh Kuppusamy, Karan Jariwala, and Deepak Sharma"

# Importing python module
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt


# Global Constants
DATA_File = 'train_data.csv'
DATA_FILE_TEST = "test_data.csv"
WEIGHTS_FILE = 'weights'

numNeuronsOutputLayer = 4
numNeuronsHiddenLayer = 5

CLASSES = 4
EPOCHS = 10000
NUM_LAYER = 2
LEARNING_RATE = 0.1


class Neuron:
    """
    Represents a neuron in the network.
    """
    __slots__ = ('n_in', 'n_out', 'weights', 'inputs', 'activation')

    def __init__(self, n_in):
        """
        Initializing parameters
        :param n_in: Number of inward connections
        :return: None
        """
        self.n_in = n_in
        self.inputs = None
        self.activation = None
        self.weights = self._init_weight()

    def _init_weight(self):
        """
        Initializing neuron weights with small random number
        :return: random weights
        """
        weights = []
        for _ in range(self.n_in):
            weights.append(random.uniform(-1, 1))
        weights = np.array(weights)
        return weights

    @staticmethod
    def _sigmoid(x):
        """
        Implementation of sigmoid activation function
        :param x: logits (weights*inputs)
        :return: sigmoid activation of neuron
        """
        return 1 / (1 + np.exp(-x))

    def _activation(self, inputs):
        """
        Computes the output activation for given features of input
        :param inputs: inputs to neuron
        :return: activation
        """
        activation = 0
        for ft in range(self.n_in):
            activation += self.weights[ft] * inputs[ft]
        return self._sigmoid(activation)

    def response(self, inputs):
        """
        Method for finding activation for a given output
        record the state of the neuron for back propagation
        :param inputs: (list) inputs to the neurons
        :return: activation
        """
        self.inputs = inputs
        self.activation = self._activation(inputs)
        return self.activation

    def get_weights(self):
        """
        Return weights of the the neuron
        :return: numpy weight array
        """
        return self.weights

    def set_weights(self, weights):
        """
        Set weights of the neuron
        :param weights: numpy array of size n_in
        :return: None
        """
        self.weights = weights


class Layer:
    """
    Represents a layer in MLP, can contain multiple neurons
    Assumption: each input to the layer is connected to each neuron of the layer
    """
    __slots__ = ('n_in', 'n_neuron', 'neurons')

    def __init__(self, n_in, n_neuron):
        """
        Initializing the layer
        :param n_in: Number of features to a neuron in the layer
        :param n_neuron: Number of neurons in the layer
        """
        self.n_in = n_in
        self.n_neuron = n_neuron
        self.neurons = self._init_neurons()

    def _init_neurons(self):
        """
        Creating and initializing the required neurons for the layer
        :return: list of neurons
        """
        neurons = []
        for _ in range(self.n_neuron):
            neurons.append(Neuron(self.n_in))
        return neurons

    def response(self, inputs):
        """
        Generating response of the layer by collecting activation of each
        neuron of the layer
        :param inputs: input list containing activation of previous/input layer
        :return: list containing response of each neuron of the layer for
        the provided inputs
        """
        response = []
        for neuron in self.neurons:
            response.append(neuron.response(inputs))
        return response

    def get_neurons(self):
        """
        Get method to return a list of neuron objects in the layer
        :return: return a list of neuron objects in the layer
        """
        return self.neurons

    def get_num_neurons(self):
        """
        Get method to return number of neurons in the layer
        :return: Number of neurons in the layer
        """
        return self.n_neuron


class MLP:
    """
    Representation of a neural network. It contain neuron layers.
    """
    __slot__ = 'network'

    def __init__(self, n_in, n_out, dims):
        """
        Creating an MLP with the given number of input and output channels
        :param n_in: Number of inputs to MLP
        :param n_out: Number of outputs from MLP
        :return: None
        """
        # layers = []
        # for i in range(len(dims) - 2):
        #     in_dim = dims[i]
        #     out_dim = dims[i + 1]
        #     layers.append(Layer(in_dim, out_dim))
        #
        # layers.append(Layer(dims[-2], dims[-1]))

        hidden_layer = Layer(n_in, numNeuronsHiddenLayer)
        output_layer = Layer(numNeuronsHiddenLayer+1, n_out)
        self.network = list([hidden_layer, output_layer])
        self.n_layers = len(self.network)

    def forward_prop(self, inputs):
        """
        Generating the activation of the MLP
        :param inputs: input to the MLP
        :return: return prediction/activation
        """
        activation = inputs
        for layer in range(self.n_layers):
            activation = self.network[layer].response(activation)
            if layer < self.n_layers-1:
                activation.insert(0, 1)
        return activation

    def network_update(self, weights):
        """
        Assign weights of the MLP
        :param weights: list(layer) of list(neuron) of np arrays(weights)
        :return: None
        """
        for layer_counter in range(self.n_layers):
            neurons = self.network[layer_counter].get_neurons()
            for neuron_counter in range(len(neurons)):
                neurons[neuron_counter].set_weights(weights[layer_counter][neuron_counter])

    def get_network_weights(self):
        """
        It returns the network weights
        :return: list(layer) of list(neuron) of np arrays(weights)
        """
        weights = []
        for layer in self.network:
            weights.append([])
            for neuron in layer.get_neurons():
                weights[-1].append(neuron.get_weights())
        return weights

    def configure_network_weight(self, weight_file):
        """
        Assign weights to the MLP's neurons provided in text file
        :param weight_file:
        :return:
        """
        file = open(weight_file, "r")
        weight_vector = file[0].strip().split(",")
        weight_counter = 0
        for layer in self.network:
            neurons = layer.get_neurons()
            for neuron in neurons:
                weight = []
                for counter in range(neuron.n_in):
                    weight.append(float(weight_vector[weight_counter]))
                    weight_counter += 1
                neuron.set_weights(np.array(weight))


def back_prop(mlp, old_weight, error):
    """
    Implementation of back propagation
    :param old_weight: list(size=layer_count) of list(size=neuron_count) of np arrays(weights)
    :param error: [true_labels]-[output layer activation]
    :param mlp: MLP
    :return: updated weight
    """
    output_layer = mlp.network[-1]  # Output layer
    output_neurons = output_layer.get_neurons()
    previous_delta = []

    for neuron_counter in range(len(output_neurons)):
        inputs = output_neurons[neuron_counter].inputs
        activation = output_neurons[neuron_counter].activation

        dsigmoid = activation * (1 - activation)  # [sig * (1 - sig) for sig in activation]
        delta = error[neuron_counter] * dsigmoid  # [err * dsig for err, dsig in zip(error, dsigmoid)]
        dw = [LEARNING_RATE * delta * inp for inp in inputs]  # should be a dot product in future
        old_weight[-1][neuron_counter] += dw
        previous_delta.append(delta)

    hidden_layer = mlp.network[-2]  # 2nd layer
    hidden_neurons = hidden_layer.get_neurons()
    output_weights = []
    hidden_delta = []

    for neuron in output_neurons:
        output_weights.append(neuron.get_weights())

    for neuron_counter in range(len(hidden_neurons)):
        inputs = hidden_neurons[neuron_counter].inputs
        activation = hidden_neurons[neuron_counter].activation

        delta = 0
        for delta_counter in range(len(previous_delta)):
            delta += previous_delta[delta_counter] * output_weights[delta_counter][neuron_counter + 1]

        hidden_delta.append(delta * activation * (1 - activation))
        dw = [LEARNING_RATE * hidden_delta[neuron_counter] * inp for inp in inputs]
        old_weight[-2][neuron_counter] += dw

    return old_weight


def load_dataset(file_name):
    """
    Read data line wise from the input file
    create attribute array with appending 1 (for bias implementation)
    looks like =[1, x1, x2]
    :param file_name:
    :return: np array of attribute and labels
    """
    data = []
    with open(file_name) as data_file:
        for line in data_file:
            line_list = line.strip().split(",")
            data.append([])
            data[-1].append(float(1))
            data[-1].append(float(line_list[0]))
            data[-1].append(float(line_list[1]))
            if float(line_list[2]) == 1.0:
                data[-1].extend([float(1), float(0), float(0), float(0)])
            if float(line_list[2]) == 2.0:
                data[-1].extend([float(0), float(1), float(0), float(0)])
            if float(line_list[2]) == 3.0:
                data[-1].extend([float(0), float(0), float(1), float(0)])
            if float(line_list[2]) == 4.0:
                data[-1].extend([float(0), float(0), float(0), float(1)])

    data = np.array(data)
    label = data[:, 3:7]
    attributes = data[:, 0:3]

    return attributes, label


def gradient_descent(network, data_file):
    """
    Implementation of Batch gradient decent algorithm
    :return: None
    """

    # Loading data
    attributes, label = load_dataset(data_file)

    # Sum of squared errors after each epoch
    sse_history = list()
    num_samples = attributes.shape[0]

    try:
        wt_file = WEIGHTS_FILE + "_" + str(EPOCHS) + ".csv"
        os.remove(wt_file)
    except OSError:
        print("No Weight file")

    for epoch in range(EPOCHS):
        sse = 0
        new_weight = network.get_network_weights()

        for sample in range(num_samples):
            prediction = network.forward_prop(attributes[sample])
            error = []
            for bit_counter in range(len(label[sample])):
                error.append(label[sample][bit_counter] - prediction[bit_counter])
            for bit_error in error:
                sse += bit_error ** 2
            new_weight = back_prop(network, new_weight, error)

        network.network_update(new_weight)

        sse_history.append(sse)
        print("After epoch " + str(epoch + 1) + "  SSE: " + str(sse))

    write_csv(network)
    return network, sse_history


def write_csv(network):
    """
    It writes the weights in a CSV file
    :param network: A neuron network
    :return: None
    """
    weight_line = ""
    weights = network.get_network_weights()
    for layer_counter in range(len(weights)):
        for neuron_counter in range(len(weights[layer_counter])):
            for weight in weights[layer_counter][neuron_counter]:
                weight_line += str(weight) + ","
    weight_line = weight_line[0:len(weight_line) - 1]
    myStr = WEIGHTS_FILE + "_" + str(EPOCHS) + ".csv"
    fp = open(myStr, "a+")
    fp.write(weight_line + "\n")
    fp.close()


def sse_vs_epoch_curve(figure, loss_matrix):
    """
    It generate a plot of  SSE vs epoch curve
    :param figure: A matplotlib object
    :param loss_matrix: A matrix loss
    :return: None
    """
    loss__curve = figure.add_subplot(111)
    loss__curve.plot(loss_matrix, label='Learning Curve')
    loss__curve.set_title("SSE vs Epochs")
    loss__curve.set_xlabel("No. Of Epochs")
    loss__curve.set_ylabel("Sum of Squared Error")
    loss__curve.legend()


def main():
    """
    Main method
    return: None
    """
    # datafile = input('provide the csv file name EX: iris.csv: ')
    datafile = "train_data.csv"
    if not os.path.isfile(datafile):
        print("Path of file name is incorrect")
        print("Usage: python3 trainMLP <input_file.csv>")
        sys.exit(1)

    network = MLP(3, 4, [3, 5, 4])
    trained_network, sse_history = gradient_descent(network, datafile)

    figure = plt.figure()
    sse_vs_epoch_curve(figure, sse_history)

    figure.show()
    plt.show()


if __name__ == "__main__":
    main()
