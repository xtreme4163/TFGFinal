import os
import numpy as np
from random import randint, uniform
from . import intelligence
from . import misfunciones as fn



class aba(intelligence.sw):
    """
    Artificial Bee Algorithm
    """

    def __init__(self, n, function, lb, ub, dimension, iteration, numeroColores ,pintor,imagen=""):
        """
        :param n: numero de individuos
        :param function: funcion
        :param lb: limite inferior del espacio
        :param ub: limite superior del espacio
        :param dimension: dimension del espacio
        :param iteration: numero de iteraciones
        """

        super(aba, self).__init__()

        #Iniciamos la poblacion y la pasamos a una lista
        self.__agents = np.random.uniform(lb, ub, (n,numeroColores, dimension))

        # Calcula el fitness de cada individuo y se coge el mejor de ellos
        fitActual = [function(x,numeroColores,imagen) for x in self.__agents]
        #Se inicia el mejor fitness con los valores iniciales de fitness
        fitMejor = fitActual 
        #Búsqueda de la mejor solucion personal de cada individuo, para devolver el individuo con una mejor solucion, para despues actualizar la mejor solución global
        Pbest = self.__agents[np.array(fitMejor).argmin()]
        Gbest = Pbest

        #Division de los individuos por grupos basandose en el numero de individuos presentes en el algoritmo
        if n <= 10:
            #Si el conjunto total de individuos es menor que 10...
            #Se divide en un grupo grande n -(n//10) ((Division entera))
            count = n - n // 2, 1, 1, 1
        else:
            #Si no se dividen en estos grupos:
            a = n // 10
            b = 5
            c = (n - a * b - a) // 2
            d = 2
            count = a, b, c, d
        
        #print("Abejas // Particulas: ",n, "Colores: ",numeroColores,"Iteraciones: ", iteration, "Imagen: ", os.path.basename(imagen))
        for t in range(iteration):
            #print("Iteración ", t+1)
            #Calculo del fitness de cada individuo
            fitness = [function(x,numeroColores, imagen) for x in self.__agents]
             # Ordenación de los índices basados en los valores de fitness
            sorted_indices = np.argsort(fitness)

            best_indices = sorted_indices[:count[0]]
            selected_indices = sorted_indices[count[0]:count[0] + count[2]]

            #Seleccion de los mejores individuos y de los otros
            best = [self.__agents[i] for i in best_indices]
            selected = [self.__agents[i] for i in selected_indices]


            #Se generan nuevos individuos a partir de los mejores y los otros
            newbee = self.__new(best, count[1], lb, ub) + self.__new(selected,
                                                                   count[3],
                                                                   lb, ub)
            m = len(newbee)
            
            # Comprobacion de si n - m es positivo antes de generar nuevas abejas aleatorias
            if n - m > 0:
                #Se actualiza la poblacion de individuos y se ajustan sus posiciones a los limites de busqueda
                additional_bees = list(np.random.uniform(lb, ub, (n - m, numeroColores, dimension)))
                self.__agents = newbee + additional_bees
            else:
                #Se actualiza la poblacion de individuos
                self.__agents = newbee[:n]

            #Se ajustan las posiciones de los individuos a los limites permitidos del espacio de busqueda
            self.__agents = np.clip(self.__agents, lb, ub)

            #Se calcula el fitnes actual de cada individuo y se guarda en una lista de tuplas junto con su indice
            fitActual = [function(x,numeroColores,imagen) for x in self.__agents]
            #Se actualiza la mejor solucion personal de cada individuo y luego la mejor solucion global
            Pbest = self.__agents[np.array(fitActual).argmin()]
            #Para cada individuo...
            for i in range(n):
                #Se comprueba si el fitness que tiene asociado es mejor que el mejor fitness guardado
                if fitActual[i] < fitMejor[i]:
                    Gbest = Pbest
                    fitMejor[i] = fitActual[i]

            #Set del mejor fitness e impresion del fitnes de la iteracion
            self.setMejorFitness(min(fitMejor))
            print(self.getMejorFitness(), end= ' ')

        ##########################################################################################################
        # Generamos la imagen cuantizada para imprimirla con el mejor valor final global.
        reducida = fn.generaCuantizada(Gbest,numeroColores, imagen)

        #print("Su fitness es: ", self.getMejorFitness())
        #Pintamos imagen
        fn.pintaImagen(reducida, imagen,pintor,"ABA",numeroColores)


    #Funcion que genera nuevos individuos para cada individuo en l moviendose a posiciones vecinas
    def __new(self, l, c, lb, ub):
        bee = []
        for i in l:
            new = [self.__neighbor(i, lb, ub) for k in range(c)]
            bee += new
        bee += l

        return bee

    #Funcion que genera un vecino aleatorio para un individuo 'who', ajustando su posicion dentro de los limites
    def __neighbor(self, who, lb, ub):

        neighbor = np.array(who) + uniform(-1, 1) * (
            np.array(who) - np.array(
                self.__agents[randint(0, len(self.__agents) - 1)]))
        neighbor = np.clip(neighbor, lb, ub)

        return list(neighbor)
