import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#define architecture of nn

data_set = np.matrix(pd.read_csv("data.csv", header=None))
labels = np.array(pd.read_csv("data_labels.csv", header=None))
test_set = np.matrix(pd.read_csv("test_set.csv", header=None))


n_in = data_set.shape[1]
n_h1 = data_set.shape[1]
n_h2 = data_set.shape[1]
n_out = 8

eta = 0.1

x_in = np.zeros((n_in, 1))
w_h1 = np.random.uniform(low=-0.1, high=0.1, size=(n_h1, n_in))
b_h1 = np.random.uniform(low=-0.1, high=0.1, size=(n_h1, 1))
w_h2 = np.random.uniform(low=-0.1, high=0.1, size=(n_h2, n_h1))
b_h2 = np.random.uniform(low=-0.1, high=0.1, size=(n_h2, 1))
w_out = np.random.uniform(low=-0.1, high=0.1, size=(n_out, n_h2))
b_out = np.random.uniform(low=-0.1, high=0.1, size=(n_out, 1))
d_out = np.zeros((n_out, 1))

Y = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]])
labs = [1, 2, 3, 4, 5, 6, 7, 8]
# m = isinstance([0, 0, 0, 0, 0, 0, 0, 1], np.array(Y))

def getAccuracy(correct, overall_data):
    accuracy = (correct / overall_data) * 100
    print("Correct predictions: ", correct)
    print("Accuracy: ", accuracy)

def plot(figure):
    plt.figure()
    plt.plot(figure)
    plt.show()


max_epoch = 30000
total_err = np.zeros((max_epoch, 1))
for q in range(max_epoch):
    # data_set = np.matrix(np.random.permutation(np.array(data_set)))
    for id, row in enumerate(data_set):
        x_in = np.transpose(row)
        d_out = np.transpose(Y[labels[id]-1])

        #forward pass
        #hidden layer 1
        v_h1 = (w_h1*x_in) + b_h1
        y_h1 = np.divide(1,(1+np.exp(-v_h1)))
        #hidden layer 2
        v_h2 = (w_h2 * y_h1) + b_h2
        y_h2 =  np.divide(1, (1 + np.exp(-v_h2)))
        #output layer
        v_out = (w_out*y_h2) + b_out
        out = np.divide(1, (1 + np.exp(-v_out)))

    #error backpropagation
        #compute error
        err = d_out - out
        #compute gradient in output layer
        delta_out = np.multiply(np.multiply(err,out), 1-out)
        #compute gradient in hidden layer 2
        delta_h2 = np.multiply(np.multiply(y_h2, 1-y_h2), np.transpose(w_out)*delta_out)
        #compute gradient in hidden layer 1
        delta_h1 = np.multiply(np.multiply(y_h1, 1-y_h1), np.transpose(w_h2)*delta_h2)
        #update weights and biases in output layer
        w_out = w_out + np.multiply(eta,delta_out)*np.transpose(y_h2)
        b_out = b_out + np.multiply(eta, delta_out)
        #update weights and biases in hidden layer 2
        w_h2 = w_h2 + np.multiply(eta, delta_h2)*np.transpose((y_h1))
        b_h2 = b_h2 + np.multiply(eta, delta_h2)

        #update weights and biases in hidden layer 1
        w_h1 = w_h1 + np.multiply(eta, delta_h1)*np.transpose((x_in))
        b_h1 = b_h1 + np.multiply(eta, delta_h1)

    total_err[q] = total_err[q] + sum(np.multiply(err, err))
    tot_end = total_err[q]
    tot_end_ind = q
    print('iteration',q, 'Error: ', total_err[q])
    #if termination condition is satisfied save weights and exit
    if(total_err[q] < 0.001):
        break

#test phase

labels_test = labels[3000:]
correct = 0
for id, row in enumerate(data_set[3000:]):
    #read data
    x_in = np.transpose(row)
    d_out = np.transpose(Y[labels_test[id]-1])
    #hidden layer 1
    v_h1 = w_h1*x_in + b_h1
    y_h1 = np.divide(1, (1 + np.exp(-v_h1)))
    #hidden layer 2
    v_h2 = w_h2*y_h1 + b_h2
    y_h2 = np.divide(1, (1 + np.exp(-v_h2)))
    #output layer
    v_out = w_out*y_h2 + b_out
    out = np.divide(1, (1 + np.exp(-v_out)))

    res = np.array([[1 if z >= 0.5 else 0 for z in (np.squeeze(np.asarray(np.transpose(out))))]])
    y_arr = [id+1 for id,x in enumerate(np.array(Y)) if (x==res).all()]
    if y_arr == np.array(labels_test[id]):
        correct = correct+1


getAccuracy(correct, len(data_set[3000:]))
plot(total_err)

plt.figure()
plt.plot(total_err)
plt.show()