import numpy as np

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

    def feedforward(self, inputs):
        layer_outputs = [inputs]
        
        for i in range(len(self.weights)):
            layer_input = np.dot(self.weights[i], layer_outputs[-1]) + self.biases[i]
            if i == len(self.weights) - 1:
                layer_output = self.sigmoid(layer_input)
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
                errors = [label - predictions]
                deltas = [errors[0] * self.sigmoid_derivative(predictions)]

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
        threshold = 0.5  # Puedes ajustar el umbral según tus necesidades
        
        # Asigna 1 si la predicción es mayor o igual al umbral, de lo contrario, asigna -1
        return np.where(predictions >= threshold, 1, -1)


if __name__ == "__main__":
    # Ejemplo de uso
    num_inputs = 2
    hidden_layers = [4]  # Una capa oculta con 4 neuronas
    num_outputs = 1
    learning_rate = 0.1
    epochs = 10000

    training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([[-1], [1], [1], [-1]])

    mlp = MultilayerPerceptron(num_inputs, hidden_layers, num_outputs, learning_rate, epochs)
    mlp.train(training_data, labels)

    for inputs in training_data:
        prediction = mlp.predict(inputs)
        print(f"Entrada: {inputs}, Predicción: {prediction}")
