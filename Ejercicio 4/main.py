import clasificador as c
if __name__ == "__main__":
    
    iris = c.Clasificador(num_inputs = 4, hidden_layers = [8], num_outputs = 3, learning_rate = 0.1, epochs = 100)
    
    iris.ReadLabels()
    iris.Entrenamiento()

    iris.Testeo()