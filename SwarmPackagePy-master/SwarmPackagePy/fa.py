from math import exp
import os
import numpy as np
import copy


from . import intelligence
from . import misfunciones as fn


class fa(intelligence.sw):
    """
    Firefly Algorithm
    
    OBSERVACION:
    Este algoritmo puede dar un mayor valor de fitness en una iteracion, haciendolo peor, creo que es debido a que la primera luciernaga se meuve libremente y no he limitado ese movimiento

    Se considera un conjunto de N luciernagas (aqui n), aqui el fitness es el brillo 
    (solucion al problema).
    Las luciernagas se atraen unas a otras, el atractivo de cada luciernada es proporcional a su brillo
    y disminuye con la distancia. La luciernaga mas brillante se mueve al azar y el resto se mueven 
    hacia la mas brillante. el brillo se ve afectado por la funcion objetivo
    
    PASOS DEL ALGORITMO:
    Generar la poblacion inicial de luciernagas
    
    REPETIR
      Mover cada luciernaga hacia las mas brillantes
      Mover la luciernaga mas brillante
      Actualizar el brillo de las luciernagas
      Ordenarlas por brillo y encontrar la mejor
    HASTA(condicion de parada)
    
    """

    def __init__(self, n, funcion, lb, ub, dimension, iteraciones,numeroColores,pintor, beta0=1, gamma=1, norm0=0, norm1=0.1,imagen=""):
        """
        :param n: numero de particulas
        :param funcion: funcion a optimizar
        :param lb: limite inferior del espacio (0 para imagenes)
        :param ub: limite superior del espacio (255 para imagenes)
        :param dimension: dimensiones del espacio
        :param iteraciones: numero de iteraciones
        :param numeroColores: numero de colores de la nueva imagen
        :param pintor: booleano que se usa para saber si pintamos imagen al final o no.
        :param beta0: atraccion mutua (Valor por defecto es 1)
        :param gamma: Coeficiente de absorcion de la luz del medio (valor por defecto 1)
        :param norm0: primer parametro para una distribucion normal (Gaussiana) (Valor por defecto 0)
        :param norm1: segundo parametro para una distribucion normal (Gaussiana) (Valor por defecto 0.1)
        :param imagen: ruta de la imagen a procesar por el algoritmo
  
        """

        super(fa, self).__init__()
        
        # Inicia la poblacion de luciernagas
        self.__agents = np.random.uniform(lb, ub, (n,numeroColores, dimension))
        
        # Calculamos el fitness de las mejores posiciones encontradas por las luciernagas
        fitnessActual = [funcion(x,numeroColores,imagen) for x in self.__agents]
        

        # Ordenar las luciérnagas por su fitness antes de empezar las iteraciones
        indicesOrdenados = np.argsort(fitnessActual)
        self.__agents = self.__agents[indicesOrdenados]
        fitnessActual = [fitnessActual[i] for i in indicesOrdenados]

        #Calculo del mejor fitness de cada iteracion y de la mejor posicion encontrada.
        fitnessMejor = funcion(self.__agents[0], numeroColores, imagen)
        Gbest =  copy.deepcopy(self.__agents[0])


        # BUCLE DEL ALGORITMO
        for t in range(iteraciones):

            # Mover la luciérnaga más brillante al azar
            self.__agents[0] += np.random.normal(norm0, norm1, (numeroColores, dimension))
            self.__agents[0] = np.clip(self.__agents[0], lb, ub)

            # Recalcular fitness para la luciérnaga que se movió al azar
            fitnessActual[0] = funcion(self.__agents[0], numeroColores, imagen)

            for i in range(1, n):
                # PARA CADA LUCIERNAGA...
                for j in range(0, i): #Se tienen en cuenta solo los individuos mas brillantes que i
                # Para cada luciernaga ...
                    if(i != j): #Comprobacion para que ningun individuo se mueva hacia si mismo
                        if fitnessActual[j] < fitnessActual[i]:
                        #Si el individuo j es mas brillante (menor fitness) que el i, se mueve i hacia j
                            self.moverLuciernaga(i, j, beta0, gamma, dimension,
                                        norm0, norm1, numeroColores)
                            self.__agents[i] = np.clip(self.__agents[i], lb, ub) #Acotar posiciones

            # Luciernagas movidas hasta aqui


            #Calculamos el fitness actual de cada luciernaga
            fitnessActual = [funcion(x,numeroColores,imagen) for x in self.__agents]


            # Ordenar las luciérnagas por su nuevo fitness
            indicesOrdenados = np.argsort(fitnessActual)
            self.__agents = copy.deepcopy(self.__agents[indicesOrdenados])
            fitnessActual = [fitnessActual[i] for i in indicesOrdenados]
            

            # Actualizar mejor solucion global
            if(fitnessActual[0] < fitnessMejor):
                fitnessMejor = fitnessActual[0]
                Gbest = copy.deepcopy(self.__agents[0])

           
            self.setMejorFitness(fitnessMejor)
            print(self.getMejorFitness(), end= ' ')
            
        Gbest = np.int_(Gbest)
        #Generamos la imagen cuantizada para imprimirla
        reducida = fn.generaCuantizada(Gbest,numeroColores,imagen)
        
        fn.pintaImagen(reducida, imagen,pintor, "FA", numeroColores)

        

    # Esta funcion mueve luciernagas ...
    def moverLuciernaga(self, i, j, beta0, gamma, dimension, norm0, norm1, numeroColores):

        # Calculo de la distancia entre dos luciernagas
        r = np.linalg.norm(self.__agents[i] - self.__agents[j]) 
        # Calculamos el ATRACTIVO de la luciernga
        # beta = atraccion mutua * exp( - coeficiente de absorcion de la luz del medio * distancia luciernagas al cuadrado)
        beta = beta0 * np.exp(-gamma * r**2)  # atractivo de la luciernaga i sobre k
        
        # Calculamos la nueva posicion aplicando esta formula 
        # fi(t+1) = fi(t) + beta(rik) * (fj(t) - fi(t)) +  aleatorio
        #Donde beta(rik) es la atraccion mutua calculada antes
        # el aleatorio se calcula usando una distribucion normal Gaussiana, entre los limites propuestos y con la 
        # dimension dicha
        self.__agents[i] = self.__agents[i] + beta * (
            self.__agents[j] - self.__agents[i]) + np.random.normal(norm0, norm1, (numeroColores, dimension))
