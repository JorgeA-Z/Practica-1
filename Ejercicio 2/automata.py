import numpy as np
import random

class Automata:
    def __init__(self, ruta):
        self.ruta = ruta

    def generatePartition(self, directorio, particiones, name):

        train = []
        label = []

        for particion in particiones:

            archivo = open(f"{directorio}{particion}", mode="r")

            lineas = archivo.readlines()
            

            for linea in lineas:
                    
                    cadenas = linea[:len(linea)-1].split(',')

                    x1 = float(cadenas[0])
                    x2 = float(cadenas[1])
                    x3 =  float(cadenas[2])
                    x4 = int(cadenas[3])

                    train.append([x1, x2, x3])
                    label.append(x4)


        random.shuffle(train)

        random.shuffle(label)   

        dataSet = open(fr"DataSets/Particiones/Particion{name}.csv", mode="w")

        for i in range(len(train)):
             dataSet.write(f"{train[i][0]},{train[i][1]},{train[i][2]},{label[i]}\n")
        
        dataSet.close()

        archivo.close()

    def getLenght(self):

        archivo = open(self.ruta, mode="r")
        
        lineas = archivo.readlines()
        lenght = 0

        for linea in lineas:
            lenght +=1

        return lenght 

    

    def dataSets(self, name):

        train = []
        label = []
        
        archivo = open(self.ruta, mode="r")

        lineas = archivo.readlines()
        

        for linea in lineas:
                
                cadenas = linea[:len(linea)-1].split(',')

                x1 = float(cadenas[0])
                x2 = float(cadenas[1])
                x3 =  float(cadenas[2])
                x4 = int(cadenas[3])

                train.append([x1, x2, x3])
                label.append(x4)


        random.shuffle(train)

        random.shuffle(label)   

        dataSet = open(fr"DataSets/Particiones/{name}.csv", mode="w")

        for i in range(len(train)):
             dataSet.write(f"{train[i][0]},{train[i][1]},{train[i][2]},{label[i]}\n")
        
        dataSet.close()

        archivo.close()
    

    def readSets(self,train, test):

        training_data = []
        training_labels = []
        test_data = []
        test_labels = []

        archivo = open(self.ruta, mode="r")
        lineas = archivo.readlines()
        
        longitud = 0

        change = False
        for linea in lineas:
                cadenas = linea[:len(linea)-1].split(',')

                x1 = float(cadenas[0])
                x2 = float(cadenas[1])
                x3 =  float(cadenas[2])
                x4 =  int(cadenas[3])
        

                if(change == False):
                    training_data.append([x1, x2, x3])
                    
                    training_labels.append(x4)
                
                else:
                    test_data.append([x1, x2, x3])
                    
                    test_labels.append(x4)
                     

                if(longitud == train):
                    change = True

                longitud+=1


        archivo.close()

        return np.array(training_data), np.array(training_labels), np.array(test_data), np.array(test_labels)

    def data(self):
        train = []
        label = []

        archivo = open(self.ruta, mode="r")
        lineas = archivo.readlines()
        for linea in lineas:
                cadenas = linea[:len(linea)-1].split(',')

                x1 = float(cadenas[0])
                x2 = float(cadenas[1])
                x3 =  int(cadenas[2])

                train.append([x1, x2])
                
                label.append(x3)
                
        archivo.close()

        return np.array(train), np.array(label)




