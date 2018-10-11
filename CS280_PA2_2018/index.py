import numpy as np
from math import exp

#define architecture of nn
n_in = 3
n_h1 = 7
n_h2 = 5
n_out = 3

eta = 0.1

x_in = np.zeros((n_in, 1))
w_h1 = np.random.uniform(low=-0.1, high=0.1, size=(n_h1, n_in))
b_h1 = np.random.uniform(low=-0.1, high=0.1, size=(n_h1, 1))
w_h2 = np.random.uniform(low=-0.1, high=0.1, size=(n_h2, n_h1))
b_h2 = np.random.uniform(low=-0.1, high=0.1, size=(n_h2, 1))
w_out = np.random.uniform(low=-0.1, high=0.1, size=(n_out, n_h2))
b_out = np.random.uniform(low=-0.1, high=0.1, size=(n_out, 1))
d_out = np.zeros((n_out, 1))
N = 8

X = np.matrix([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])
Y = np.matrix([[0, 0, 0],
              [1, 1, 0],
              [1, 0, 1],
              [0, 1, 1],
              [0, 1, 1],
              [1, 0, 0],
              [1, 1, 0],
              [0, 0, 0]])

max_epoch = 30000
total_err = np.zeros((max_epoch, 1))

for q in range(max_epoch):
    # p = np.random.permutation(range(1,N+1))
    p = np.random.permutation(N)
    for n in p:
        x_in = np.transpose(X[n])
        d_out = np.transpose(Y[n])
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
    if(q%500 == 0):
        print('iteration',q, 'Error: ', total_err[q])
    #if termination condition is satisfied save weights and exit
    if(total_err[q] < 0.001):
        break

#test phase
nn_output = np.zeros(Y.shape)
for n in range(N):
    #read data
    x_in = np.transpose(X[n])
    d_out = np.transpose((Y[n]))
    #hidden layer 1
    v_h1 = w_h1*x_in + b_h1
    y_h1 = np.divide(1, (1 + np.exp(-v_h1)))
    #hidden layer 2
    v_h2 = w_h2*y_h1 + b_h2
    y_h2 = np.divide(1, (1 + np.exp(-v_h2)))
    #output layer
    v_out = w_out*y_h2 + b_out
    out = np.divide(1, (1 + np.exp(-v_out)))








