import os
import numpy as np

from . import intelligence
from . import misfunciones as fn



class gwo(intelligence.sw):
    """
    Grey Wolf Optimizer
    """

    def __init__(self, n, function, lb, ub, dimension, iteration, numeroColores ,pintor,imagen=""):
        """
        :param n: numero de individuos
        :param function: funcion del algoritmo
        :param lb: limite inferior del espacio de busqueda
        :param ub: limite superior del espacio de busqueda
        :param dimension: dimension del espacio
        :param iteration: numero de iteraciones
        """

        super(gwo, self).__init__()

        #Inicio de la poblacion
        self.__agents = np.random.uniform(lb, ub, (n,numeroColores, dimension))

        #Calculo de los valores de fitness de los individuos
        fitnessA= [function(x,numeroColores,imagen) for x in self.__agents]
        #Buscamos los mejores lobos
        alpha, beta, delta, fitActual = self.getABD(n, fitnessA)
        #Seteamos el valor del mejor fitnes con el valor de fitActual (valor del fitness del lobo alpha)
        fitMejor = fitActual
        Gbest = alpha

        #print("GWO // Particulas: ",n, "Colores: ", numeroColores,"Iteraciones: ", iteration, "Imagen: ", os.path.basename(imagen))
        for t in range(iteration):
            #print("Iteración ", t+1)
            #Actualizo el parámetro del algoritmo (a)
            a = 2 - 2 * t / iteration

            #Cálculo de los vectores aleatorios entre [0, 1], y de A y C para lobo alpha
            r1 = np.random.rand(n,numeroColores,dimension)
            r2 = np.random.rand(n,numeroColores,dimension)
            A1 = 2 * r1 * a - a
            C1 = 2 * r2

            #Cálculo de los vectores aleatorios entre [0, 1], y de A y C para lobo beta
            r1 = np.random.rand(n,numeroColores,dimension)
            r2 = np.random.rand(n,numeroColores,dimension)
            A2 = 2 * r1 * a - a
            C2 = 2 * r2


            #Cálculo de los vectores aleatorios entre [0, 1], y de A y C para lobo delta
            r1 = np.random.rand(n,numeroColores,dimension)
            r2 = np.random.rand(n,numeroColores,dimension)
            A3 = 2 * r1 * a - a
            C3 = 2 * r2

            #Cálculo de D (Para los lobos alpha, beta y delta)
            Dalpha = abs(C1 * alpha - self.__agents) 
            Dbeta = abs(C2 * beta - self.__agents)
            Ddelta = abs(C3 * delta - self.__agents)

            #Cálculo de X para la nueva posicion
            X1 = alpha - A1 * Dalpha
            X2 = beta - A2 * Dbeta
            X3 = delta - A3 * Ddelta

            #Nueva posicion de cada lobo
            self.__agents = (X1 + X2 + X3) / 3

            #Ajuste de estas nuevas posiciones al limite del espacio
            self.__agents = np.clip(self.__agents, lb, ub)

            
            #Se calcula el fitness actual de cada individuo
            fitnessA= [function(x,numeroColores,imagen) for x in self.__agents]
            #Cálculo de los lobos alfa, beta y delta
            alpha, beta, delta, fitActual = self.getABD(n, fitnessA)

            #Cálculo de Gbest (Mejor solucion) y actualizacion del valor del mejor fitness
            if fitActual < fitMejor:
                Gbest = alpha
                fitMejor = fitActual
                
            #Conseguimos el mejor fitness y lo mostramos en pantalla
            self.setMejorFitness(fitMejor)
            print(self.getMejorFitness(), end= ' ')

  
        #Generamos la cuantizada para imprimirla junto al valor final del algoritmo.
        reducida = fn.generaCuantizada(Gbest,  numeroColores,imagen)
        #print("Fitness final --> ", self.getMejorFitness())
        fn.pintaImagen(reducida, imagen,pintor,"GWO",numeroColores)


    """ Funcion que devuelve los lobos alfa beta y delta y el fitnes del mejor lobo"""
    def getABD(self, n,fitnessA):

        result = []
        # Calcula el fitness de cada agente y los guarda junto con su índice en una lista de tuplas
        fitness = [(fitnessA[i], i) for i in range(n)]
        # Ordena la lista de fitness en orden ascendente (menor fitness es mejor)
        fitness.sort()
        # Selecciona los tres agentes con mejor fitness
        for i in range(3):
            result.append(self.__agents[fitness[i][1]])
        result.append(fitness[0][0])
        return result

    def get_leaders(self):
        """Return alpha, beta, delta leaders of grey wolfs"""

        return list(self.__leaders)
