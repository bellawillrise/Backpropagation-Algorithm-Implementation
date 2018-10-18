import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def read_data():
    data_set = np.matrix(pd.read_csv("data.csv", header=None))
    labels = np.matrix(pd.read_csv("data_labels.csv", header=None))
    test_set = np.matrix(pd.read_csv("test_set.csv", header=None))
    return data_set, labels, test_set

def initialize_network(inputlayer_neurons, hiddenlayer1_neurons, hiddenlayer2_neurons, output_neurons):
    weights_hiddenlayer1 = np.random.uniform(low=-0.1, high=0.1, size=(hiddenlayer1_neurons, inputlayer_neurons))
    biases_hiddenlayer1 = np.random.uniform(low=-0.1, high=0.1, size=(hiddenlayer1_neurons, 1))
    weights_hiddenlayer2 = np.random.uniform(low=-0.1, high=0.1, size=(hiddenlayer2_neurons, hiddenlayer1_neurons))
    biases_hiddenlayer2 = np.random.uniform(low=-0.1, high=0.1, size=(hiddenlayer2_neurons, 1))
    weights_output = np.random.uniform(low=-0.1, high=0.1, size=(output_neurons, hiddenlayer2_neurons))
    biases_output = np.random.uniform(low=-0.1, high=0.1, size=(output_neurons, 1))
    return weights_hiddenlayer1, biases_hiddenlayer1, weights_hiddenlayer2, biases_hiddenlayer2, weights_output, biases_output

def initialize_neurons(data_set):
    inputlayer_neurons = data_set.shape[1]
    hiddenlayer1_neurons = data_set.shape[1]
    hiddenlayer2_neurons = data_set.shape[1]
    output_neurons = 1
    return (inputlayer_neurons, hiddenlayer1_neurons, hiddenlayer2_neurons, output_neurons)

def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

def sigmoidPrime(sigmoid):
    return np.multiply(sigmoid,(1-sigmoid))

def getAccuracy(correct, overall_data):
    accuracy = (correct / overall_data) * 100
    print("Correct predictions: ", correct)
    print("Accuracy: ", accuracy)

def plot(figure):
    plt.figure()
    plt.plot(figure)
    plt.show()


def trainNN(training_set, labels):
    inputlayer_neurons, hiddenlayer1_neurons, hiddenlayer2_neurons, output_neurons = initialize_neurons(training_set)
    weights_hiddenlayer1, biases_hiddenlayer1, weights_hiddenlayer2, biases_hiddenlayer2, weights_output, biases_output = initialize_network(inputlayer_neurons, hiddenlayer1_neurons, hiddenlayer2_neurons, output_neurons)

    eta = 0.1
    max_epoch = 100
    total_err = np.zeros((max_epoch, 1))
    labels = labels / 10
    for i in range((max_epoch)):
        for id, row in enumerate(training_set):
            #forward propagate
            v_h1 = (weights_hiddenlayer1 * np.transpose(row)) + biases_hiddenlayer1
            y_h1 = sigmoid(v_h1)
            v_h2 = (weights_hiddenlayer2 * y_h1) + biases_hiddenlayer2
            y_h2 = sigmoid(v_h2)
            v_out = (weights_output * y_h2) + biases_output
            out = sigmoid(v_out)
            # compute error
            err = labels[id] - out
            #compute delta
            delta_out = np.multiply(err, sigmoidPrime(out))
            delta_h2 = np.multiply(sigmoidPrime(y_h2), np.transpose(weights_output) * delta_out)
            delta_h1 = np.multiply(sigmoidPrime(y_h1), np.transpose(weights_hiddenlayer2) * delta_h2)

            #backpropagate
            weights_output = weights_output + eta * np.multiply(delta_out, np.transpose(y_h2))
            biases_output = biases_output + (eta * delta_out)
            weights_hiddenlayer2 = weights_hiddenlayer2 + eta * np.multiply(delta_h2, np.transpose((y_h1)))
            biases_hiddenlayer2 = biases_hiddenlayer2 + (eta * delta_h2)
            weights_hiddenlayer1 = weights_hiddenlayer1 + eta * np.multiply(delta_h1, np.transpose(row))
            biases_hiddenlayer1 = biases_hiddenlayer1 + (eta * delta_h1)

        total_err[i] = total_err[i] + sum(np.multiply(err, err))
        print('iteration', i, 'Error: ', total_err[i])

        if (total_err[i] < 0.001):
            break

    # test phase
    labels_test = labels[3000:]
    correct = 0
    for id, row in enumerate(training_set[3000:]):
        v_h1 = weights_hiddenlayer1 * np.transpose(row) + biases_hiddenlayer1
        y_h1 = sigmoid(v_h1)
        v_h2 = weights_hiddenlayer2 * y_h1 + biases_hiddenlayer2
        y_h2 = sigmoid(v_h2)
        v_out = weights_output * y_h2 + biases_output
        out = sigmoid(v_out)

        if (np.floor(out * 10) == np.floor(labels_test[id] * 10)):
            correct = correct + 1

    getAccuracy(correct, len(training_set[3000:]))
    plot(total_err)


if __name__ == '__main__':
    training_set, labels, test_set = read_data()
    trainNN(training_set, labels)