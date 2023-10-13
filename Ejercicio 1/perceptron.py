import numpy as np

class perceptron:
    def __init__(self, num_inputs, learning_rate, epochs):
        self.weights = np.zeros(num_inputs)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        # Calcula el producto punto entre los pesos y las entradas y agrega el sesgo.
        activation = np.dot(self.weights, inputs) + self.bias
        # La función de activación es una función umbral simple (0 si es negativo, 1 si es positivo).
        return -1 if activation < 0 else 1

    def train(self, training_data, labels):
        for i in range(self.epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                # Actualiza los pesos y el sesgo según el error.
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)

    def getW(self):
        return self.weights
    
    def getb(self):
        return self.bias

if __name__ == "__main__":
    pass
