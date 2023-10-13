import perceptronM as p
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import LeavePOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class Clasificador:

    def __init__(self, num_inputs, hidden_layers, num_outputs, learning_rate, epochs):
        
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        #Labels de entrenamiento y testeo
        self.training_data = None
        self.training_labels = None
        self.test_data = None
        self.test_labels = None

        #Ruta de extracción de datos
        self.ruta = 'irisbin.csv'

        #Perceptron multicapa
        self.Perceptron = p.MultilayerPerceptron(self.num_inputs, self.hidden_layers, self.num_outputs, self.learning_rate, self.epochs);

    def ReadLabels(self):
        #Arrays provisionales de entrenamiento y testeo
        data = []
        labels = []
        
        archivo = open(self.ruta, mode='r')
        
        lineas = archivo.readlines()

        #Extracción de los datos de testeo y entrenamiento
        for linea in lineas:
            cadenas = linea[:len(linea)-1].split(',')
            
            x1 = float(cadenas[0])
            x2 = float(cadenas[1])
            x3 =  float(cadenas[2])
            x4 =  float(cadenas[3])

            y1 = int(cadenas[4])
            y2 =  int(cadenas[5])
            y3 =  int(cadenas[6])

            data.append([x1, x2, x3, x4])
            
            labels.append([y1, y2, y3])
        

        archivo.close()
        self.training_data, self.training_labels, self.test_data, self.test_labels = self.PerturbateLabels(data, labels)
    



    def PerturbateLabels(self, data, labels):

        x = np.array(data)
        y = np.array(labels)

        # Normalize the features
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        #Perturbación del perceptron 80/20
        porcentaje = 0.8

        num_elementos_array2 = int(len(data) * porcentaje)

        #Division de los arrays
        trainData = np.array(x[:num_elementos_array2])
        
        testData = np.array(x[num_elementos_array2:])

        trainLabels = np.array(y[:num_elementos_array2])

        testLabels = np.array(y[num_elementos_array2:])

        return trainData, trainLabels, testData, testLabels



    def Entrenamiento(self):
        #Entrenamiento del perceptron
        self.Perceptron.train(self.training_data, self.training_labels)

    def Tipo(self, prediccion, real):

        if(prediccion[0] ==  real[0] and prediccion[1] ==  real[1] and prediccion[2] ==  real[2]):

            if(prediccion[0] == -1 and prediccion[1] == -1 and prediccion[2] == 1):
                return "Setosa"
            elif(prediccion[0] == -1 and prediccion[1] == 1 and prediccion[2] == -1):
                return "Versicolor"
            elif(prediccion[0] == 1 and prediccion[1] == -1 and prediccion[2] == -1):
                return "Virginica"
        else:
            return "None"

    def leav_one_out(self, pe = 1):
        x, y = self.training_data, self.training_labels

        accuracies = []
        lpo = LeavePOut(pe)

        for train_indices, test_indices in lpo.split(x):
            X_train, X_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Inicializa y entrena el modelo en el conjunto de entrenamiento
            model = p.MultilayerPerceptron(self.num_inputs, self.hidden_layers, self.num_outputs, self.learning_rate, self.epochs);

            model.train(X_train, y_train)
            predictions = []
            for inputs in X_test:
                prediction = model.predict(inputs)
                predictions.append(prediction)

            # Calcula la precisión y almacénala
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)


        average_accuracy = sum(accuracies) / len(accuracies)
        
        std_deviation = np.std(accuracies)
        
        error_esperado_porcentaje = (1 - average_accuracy) * 100



        if(pe > 1):
            print("Leave-K-Out")
        else:
            print("Leave-One-Out")

        print(f"Precisión en el conjunto de prueba: {average_accuracy * 100:.2f}%")
        
        print(f"Desviación Estándar: {std_deviation * 100:.2f}%")

        print(f"Error Esperado: {error_esperado_porcentaje}%")



    def Testeo(self):
        
        correct_predictions = 0
        total_predictions = len(self.test_data)

        predicted_labels = []
        
        # Define las etiquetas para cada clase
        colors = ['red', 'green', 'blue']
        labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        
        for inputs, label in zip(self.test_data, self.test_labels):
            prediction = self.Perceptron.predict(inputs)
            predicted_labels.append(prediction)
            
            print(f"Entradas: {inputs}, Real: {label}, Predicción: {prediction}, Tipo: {self.Tipo(prediction, label)}")
            

            if self.Tipo(prediction, label) != "None":
                plt.scatter(inputs[0], inputs[1], color=colors[np.argmax(label)])
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions

        print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")
        
        self.leav_one_out()

        self.leav_one_out(2)



        for i, label in enumerate(labels):
            plt.scatter([], [], color=colors[i], label=label)
        plt.legend()

        plt.xlabel('Longitud del Sépalo (cm)')
        plt.ylabel('Longitud del Pétalo (cm)')
        plt.title('Distribución de clases en el dataset')
        plt.show()
