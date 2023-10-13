import numpy as np
import perceptron
import automata
import matplotlib.pyplot as plt
import perceptronM as m

def EntrenamientoSimple():
    a = automata.Automata("DataSets/OR_trn.csv");
    b = automata.Automata("DataSets/OR_tst.csv");

    # Datos de entrenamiento y prueba (XOR)
    training_data, training_labels = a.data()
    test_data, test_labels = b.data()

    # Crea un perceptrón con 2 entradas (para el operador XOR)
    p = perceptron.perceptron(num_inputs=2, learning_rate=0.1, epochs=100)

    # Entrena el perceptrón
    p.train(training_data, training_labels)

    # Prueba el perceptrón
    correct_predictions = 0
    total_predictions = len(test_data)

    predicted_labels = []

    for inputs, label in zip(test_data, test_labels):
        prediction = p.predict(inputs)
        predicted_labels.append(prediction)
        print(f"Entradas: {inputs}, Predicción: {prediction}, Real: {label}")
        if prediction == label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")


    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    # Crear una malla de puntos para la visualización
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    mesh_data = np.c_[xx.ravel(), yy.ravel()]

    # Calcular las predicciones del perceptrón en la malla de puntos
    Z = np.array([p.predict(point) for point in mesh_data])
    Z = Z.reshape(xx.shape)

    # Visualizar los datos de prueba
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.6)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, marker='o', s=25)
    plt.title("Agrupación de datos y Frontera de Decisión")
    plt.show()




def EntrenamientoMulticapa():
    a = automata.Automata("DataSets/XOR_trn.csv");
    b = automata.Automata("DataSets/XOR_tst.csv");

    # Datos de entrenamiento y prueba (XOR)
    training_data, training_labels = a.data()
    test_data, test_labels = b.data()

    p = m.MultilayerPerceptron(num_inputs = 2, hidden_layers = [4], num_outputs = 1, learning_rate = 0.1, epochs = 100)
   
    p.train(training_data, training_labels)

    # Prueba el perceptrón
    correct_predictions = 0
    total_predictions = len(test_data)

    predicted_labels = []

    for inputs, label in zip(test_data, test_labels):

        prediction = p.predict(inputs)
        
        print(f"Entradas: {inputs}, Predicción: {prediction}, Real: {label}")
        predicted_labels.append(prediction)

        if prediction == label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")


    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    # Crear una malla de puntos para la visualización
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    mesh_data = np.c_[xx.ravel(), yy.ravel()]

    # Calcular las predicciones del perceptrón en la malla de puntos
    Z = np.array([p.predict(point) for point in mesh_data])
    Z = Z.reshape(xx.shape)

    # Visualizar los datos de prueba
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.6)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, marker='o', s=25)
    plt.title("Agrupación de datos y Frontera de Decisión")
    plt.show()


def EntrenamientoPerturbado():
    llave = True
    particiones = int()
    prueba = int()
    entrenamiento = int()
    longitud = int()


    while llave:
        print("|Entrenamiento Perturbado|")
        try:
            particiones = int(input("Ingresa el número de particiones: \n"))
            entrenamiento = int(input("Ingresa el porcentaje de entrenamiento: \n"))
            prueba = int(input("Ingresa el porcentaje de prueba: \n"))

        except Exception as e:
            print(f"Error: {e}")
        else:
            llave = False
    

    archivos = ["DataSets/spheres2d10.csv", "DataSets/spheres2d50.csv", "DataSets/spheres2d70.csv"]
    p = []

    for i in range(particiones):
        preparticiones = [] 
        for archivo in archivos:
            a = automata.Automata(archivo)

            a.dataSets(f"{archivo[9:20]}_{i}")

            preparticiones.append(f"{archivo[9:20]}_{i}.csv")
            
        
        
        a.generatePartition("DataSets/Particiones/",preparticiones, i)
        
        p.append(f"DataSets/Particiones/Particion{i}.csv")
        
    
    print(p)  

    testPerturbado(p, prueba, entrenamiento)


    


def testPerturbado(particiones, prueba, entrenamiento):


    for particion in particiones:
        print(f"\n\n\n\nEntrenamiento {particion}\n\n\n\n")
        
        b = automata.Automata(f"{particion}")

        longitud = b.getLenght()
    
        test = (longitud*prueba) / 100
        
        train = (entrenamiento*longitud) / 100 

        training_data, training_labels, test_data, test_labels = b.readSets(train, test)
        

        # Crea un perceptrón con 3 entradas
        p = perceptron.perceptron(num_inputs=3, learning_rate=0.1, epochs=100)

        # Entrena el perceptrón
        p.train(training_data, training_labels)

        # Prueba el perceptrón
        correct_predictions = 0
        total_predictions = len(test_data)

        predicted_labels = []

        for inputs, label in zip(test_data, test_labels):
            prediction = p.predict(inputs)
            predicted_labels.append(prediction)
            print(f"Entradas: {inputs}, Predicción: {prediction}, Real: {label}")
            if prediction == label:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=test_labels, marker='o')
        plt.title(f"Entrenamiento {particion}")
        plt.show()

def MainUserInterface():
    llave = True

    while llave:
        print("|Entrenamiento con Perceptron|")
        print("|1. Entrenamiento Simple     |")
        print("|2. Entrenamiento Multicapa  |")
        print("|3. Entrenamiento Perturbado |")
        print("|4. Salir                    |")

        try:
            op = int(input("Ingresa la opcion: "))

        except Exception as e:
            print(f"Error: {e}")

        else:

            if(op == 1):
                EntrenamientoSimple()

            elif(op == 2):
                EntrenamientoMulticapa()

            elif(op == 3):
                EntrenamientoPerturbado()
            
            elif(op == 4):
                print("Hasta pronto!")
                llave = False

            else:
                print("Opcion invalida")
            

if __name__ == "__main__":
    MainUserInterface();
   
    