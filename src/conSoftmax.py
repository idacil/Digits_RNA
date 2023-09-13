#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        #AQUÍ DEFINO NUEVAS VARIABLES m y v PARA EL OPTIMIZADOR Adam 
        self.me = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        self.ve = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        
        self.me0 = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        self.ve0 = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def Adam(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):        
        
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        
        
        #EN ESTE RENGLÓN COLOCO LA ESPRESIÓN PARA EL PARÁMETRO g
        beta1 = 0.1
        beta2 = 0.01
        eps = 0.0000000004
        
        #QUÍ SUSTITUIMOS POR LOS NUEVOS PARÁMETROS M y V
        self.me = [(beta1*m + (1 - beta1)*nw)
                  for m, nw in zip(self.me, nabla_w)]
        
        self.ve = [(beta2*v + (1 - beta2)*np.sum(nw**2))
                  for v, nw in zip(self.ve, nabla_w)]
        
        self.me0 = [m/(1-beta1)
                  for m in self.me]
        
        self.ve0 = [v/(1-beta1)
                  for v in self.ve]
     
        
        self.weights = [w-(eta/(len(mini_batch)*v0 + eps))*m0
                        for w, m0, v0 in zip(self.weights, self.me0, self.ve0)]
        
    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #AQUÍ INTENTÉ COLOCAR LA FUNCIÓN PARA CROSS ENTROPY
    def cost_derivative(self, output_activations, y):
        y_true = y
        y_pred = output_activations
        y_pred = sigmoid(y_pred)
        loss = 0
        for i in range(len(y_pred)):
            loss = loss + (-1 * y_true[i]*np.log(y_pred[i]))
        return loss
    

# AQUÍ INTENTÉ COLOCO LA FUNCIÓN SOFTMAX
def sigmoid(z):
    return np.exp(z)/np.sum(np.exp(z))

#def sigmoid(z):
#    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
