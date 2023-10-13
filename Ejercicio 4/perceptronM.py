import numpy as np
import matplotlib.pyplot as plt

class MultilayerPerceptron:
    def __init__(self, num_inputs, hidden_layers, num_outputs, learning_rate, epochs):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Inicializar pesos y sesgos para capas ocultas y de salida
        self.weights = [np.random.rand(hidden_layers[0], num_inputs)]
        self.biases = [np.zeros(hidden_layers[0])]
        
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.rand(hidden_layers[i], hidden_layers[i-1]))
            self.biases.append(np.zeros(hidden_layers[i]))
        
        self.weights.append(np.random.rand(num_outputs, hidden_layers[-1]))
        self.biases.append(np.zeros(num_outputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def softmax_derivative(self, s):
        return s * (1 - s)

    def feedforward(self, inputs):
        layer_outputs = [inputs]
        for i in range(len(self.weights)):
            layer_input = np.dot(self.weights[i], layer_outputs[-1]) + self.biases[i]
            if i == len(self.weights) - 1:
                layer_output = self.softmax(layer_input)  # Softmax activation for output layer
            else:
                layer_output = self.relu(layer_input)  # ReLU activation for hidden layers
            layer_outputs.append(layer_output)
        return layer_outputs


    def train(self, training_data, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_data, labels):
                # Forward propagation
                layer_outputs = self.feedforward(inputs)
                predictions = layer_outputs[-1]

                # Backpropagation
                # Cross-entropy error calculation
                errors = [-label * np.log(predictions)]
                deltas = [errors[0] * self.softmax_derivative(predictions)]

                # Calculate errors and deltas for hidden layers
                for i in range(len(self.weights) - 2, -1, -1):
                    error = deltas[0].dot(self.weights[i + 1])
                    errors.insert(0, error)
                    deltas.insert(0, errors[0] * self.relu_derivative(layer_outputs[i + 1]))

                # Update weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * np.outer(deltas[i], layer_outputs[i])
                    self.biases[i] += self.learning_rate * deltas[i]

                    
    def predict(self, inputs):
        layer_outputs = self.feedforward(inputs)
        predictions = layer_outputs[-1]
        # Create a binary array where 1 represents the class with the highest probability and -1 represents all other classes
        return np.where(predictions == np.max(predictions), 1, -1)
