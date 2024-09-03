import os
import numpy as np

from . import intelligence
from . import misfunciones as fn


class ballena(intelligence.sw):
    """
    Whale Swarm Algorithm
    """

    def __init__(self, n, function, lb, ub, dimension, iteration,numeroColores,pintor, ro0=2,
                 eta=0.005,imagen=""):
        """
        Se supone que la mejor solucion candidata actual se acerca a la presa objetivo
        y otras soluciones actualizan su posicion hacia la mejor ballena
        
       
        :param n: numero de ballenas
        :param function: funcion a optimizar
        :param lb: limite inferior del espacio
        :param ub: limite superior del espacio
        :param dimension: dimension del espacio
        :param iteration: numero de iteraciones
        :param numeroColores: numero de colores de la nueva imagen
        :param pintor: booleano que se usa para saber si pintamos imagen al final o no.
        :param ro0: intensidad de ultrasonido en la fuente de origen (default value is 2)
        :param eta: probabilidad de distorsion de mensaje a largas distancias (default value is 0.005) 
        :param imagen: ruta de la imagen a procesar por el algoritmo

        """
        super(ballena, self).__init__()


        # Inicializar la poblacion de ballenas
        self.__agents = np.random.uniform(lb, ub, (n, numeroColores, dimension))
        self._points(self.__agents)
        
        # Inicializamos el vector de las mejores posiciones de las ballenas
        Pbest = self.__agents
        
        # Vectores con el fitness actual, y el mejor fitness hallado.
        fitActual = [function(x,numeroColores,imagen) for x in self.__agents] 
        fitMejor = fitActual   # Fitness de la iteracion actual                         
        # Cogemos la mejor solucion global
        #Gbest = Pbest

        #print("WSA // Particulas: ",n, "Colores: ",numeroColores,"Iteraciones: ", iteration, "Imagen: ", os.path.basename(imagen))
        # Algoritmo, se repite hasta el nº de iteracciones
        for t in range(iteration):
            
            
            #print("Iteración ", t+1)
            new_agents = self.__agents
            
            #Para cada particula ...
            for i in range(n):
                #Buscamos una ballena con mejor fitness y que sea la mas cercana
                y = self.__better_and_nearest_whale(i, n, fitActual, fitMejor)
                # Si hemos encontrado otra ballena ...
                if y:
                # Movemos a la ballena i hacia otra mejor y cercana (depende de los dos parametros del algoritmo)
                    new_agents[i] += np.dot(
                        np.random.uniform(0, ro0 *
                            np.exp(-eta * self.__whale_dist(i, y))),
                        self.__agents[y] - self.__agents[i])
            # Movemos Ballenas
            self.__agents = new_agents
            # Acotamos al espacio de solucion
            self.__agents = np.clip(self.__agents, lb, ub)
            self._points(self.__agents)
            
            
            #Actualizamos su fitness actual
            fitActual = [function(x,numeroColores,imagen) for x in self.__agents]
            
            #Actualizamos la mejor solucion particular
            
            # Para cada particula ...
            for i in range(n):
               # Si el fitness de esta posicion es menor que el almacenado lo actualizamos
               #if(function(self.__agents[i],r) < function(Pbest[i],r)):
               if(fitActual[i] < fitMejor[i]):   
                  Pbest[i] = self.__agents[i]
                  fitMejor[i] = fitActual[i]
            
            #Actualizamos la mejor solucion global
            Gbest = Pbest[np.array([function(x,numeroColores,imagen) for x in Pbest]).argmin()] 
            
            self.setMejorFitness(function(Gbest,numeroColores,imagen))
            print(self.getMejorFitness(), end= ' ')
            
        Gbest= np.int_(Gbest)
        self._set_Gbest(Gbest)

        # Generamos la imagen cuantizada para pintarla
        reducida = fn.generaCuantizada(Gbest,numeroColores,imagen)
        
        #print("Fitness final: ", self.getMejorFitness())
        #Pintamos la imagen
        fn.pintaImagen(reducida, imagen,pintor,"BA",numeroColores)

        
        

    """
    Funcion que devuelve la distancia entre dos vectores. En este caso
    devuelve la distancia entre dos ballenas
    Argumentos:
      i: ballena i
      j: ballena j
    """
    def __whale_dist(self, i, j):
        # Calculamos la distancia entre las ballenas, restando el vector ... 
        return np.linalg.norm(self.__agents[i] - self.__agents[j])


    """
    Funcion que compara una ballena con el resto, y devuelve la ballena mas cercana que tenga un mejor fitness 
    Argumentos: 
     u: numero de particula pasada.
     n: numero de ballenas
     function. funcion a optimizar
    """
    def __better_and_nearest_whale(self, u, n, fitActual, fitMejor):
        
        # Usado para comparar y quedarnos con la menor distancia
        temp = float("inf")

        v = None
        
        for i in range(n):
        #Para todas las ballenas ...
           
            if fitActual[i] < fitActual[u]:
            # Si el fitness de i es menor que el fitness de u ...
                # la distancia de i a u es 
                dist_iu = self.__whale_dist(i, u)
                if dist_iu < temp:
                    v = i
                    temp = dist_iu
        return v #Devolvemos ballena
