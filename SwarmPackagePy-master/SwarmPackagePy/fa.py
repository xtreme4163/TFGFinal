from math import exp
import os
import numpy as np

from . import intelligence
from . import misfunciones as fn


class fa(intelligence.sw):
    """
    Firefly Algorithm
    
    Poner formula matemática en mover luciernaga
    
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

    def __init__(self, n, function, lb, ub, dimension, iteration,numeroColores,pintor, csi=1, psi=1,
                 alpha0=1, alpha1=0.1, norm0=0, norm1=0.1,imagen=""):
        """
        :param n: numero de particulas
        :param function: funcion a optimizar
        :param lb: limite inferior del espacio (0 para imagenes)
        :param ub: limite superior del espacio (255 para imagenes)
        :param dimension: dimensiones del espacio
        :param iteration: numero de iteraciones
        :param numeroColores: numero de colores de la nueva imagen
        :param pintor: booleano que se usa para saber si pintamos imagen al final o no.
        :param csi: atraccion mutua (Valor por defecto es 1)
        :param psi: Coeficiente de absorcion de la luz del medio (valor por defecto 1)
        :param alpha0: valor inicial del parametro aleatorio alpha (valor por defecto 1) 
        :param alpha1: valor final del parametro aleatorio alpha (valor por defecto 0.1)
        :param norm0: primer parametro para una distribucion normal (Gaussiana) (Valor por defecto 0)
        :param norm1: segundo parametro para una distribucion normal (Gaussiana) (Valor por defecto 0.1)
        :param imagen: ruta de la imagen a procesar por el algoritmo
  
        """

        super(fa, self).__init__()
        
        # Inicia la poblacion de luciernagas
        self.__agents = np.random.uniform(lb, ub, (n,numeroColores, dimension))
        self._points(self.__agents)

        # Iniciamos las mejores posiciones de cada particula con las calculadas antes
        Pbest = self.__agents
        
        # Calculamos el fitness de las mejores posiciones encontradas por las luciernagas
        fitnessP = [function(x,numeroColores,imagen) for x in self.__agents]
        fitnessA = fitnessP # Lo igualamos al de la posicion ACTUAL
   
        print("Luciernagas // Particulas: ",n, "Colores: ",numeroColores,"Iteraciones: ", iteration, "Imagen: ", os.path.basename(imagen))
        # BUCLE DEL ALGORITMO
        for t in range(iteration):
            print("Iteración ", t+1)
            # Esto se usa en la funcion mover para el calculo de un numero aleatorio
            alpha = alpha1 + (alpha0 - alpha1) * exp(-t)

            for i in range(n):
                # PARA CADA LUCIERNAGA...
                
                for j in range(n):
                # Para cada particula ...
                    if fitnessA[i] > fitnessA[j]:
                    # Si el fitness de la particula i es mayor que la de la j
                    # entonces la movemos hacia la más brillante
                        self.__move(i, j, t, csi, psi, alpha, dimension,
                                    norm0, norm1)
                    else:
                    # Si es menor entonces la movemos al azar, le sumamos un numero aleatorio
                        self.__agents[i] += np.random.normal(norm0, norm1,
                                                             dimension)
            # Acotamos la posicion a nuestro rango permitido
            self.__agents = np.clip(self.__agents, lb, ub)
            self._points(self.__agents)
            # Luciernagas movidas hasta aqui
            #Calculamos el fitness actual de cada luciernaga
            fitnessA = [function(x,numeroColores,imagen) for x in self.__agents]
            
            
            
            # Actualizar la mejor solucion particular
            # Si el fitness de la posicion actual es menor (mejor) que el mejor guardado ...
            for i in range(n):
            #Para cada luciernaga
              if(fitnessA[i] < fitnessP[i]):
              # Actualizamos su posicion y su mejor fitness
                  Pbest[i] = self.__agents[i] # posicion
                  fitnessP[i] = function(Pbest[i],numeroColores,imagen) # fitness
                  
              
            
            # Actualizar mejor solucion global
            Gbest = Pbest[np.array([function(x,numeroColores,imagen) for x in Pbest]).argmin()]
            
            self.setMejorFitness(function(Gbest,numeroColores,imagen))
            print("Fitness --> ", self.getMejorFitness())
            
        Gbest = np.int_(Gbest)
        self._set_Gbest(Gbest)
        
        #Generamos la imagen cuantizada para imprimirla
        reducida = fn.generaCuantizada(Gbest,numeroColores,imagen)
        
        print("Fitness final --> ", self.getMejorFitness())
        fn.pintaImagen(reducida, imagen,pintor)

        

    # Esta funcion mueve luciernagas ...
    def __move(self, i, j, t, csi, psi, alpha, dimension, norm0, norm1):

        # Calculo de la distancia entre dos luciernagas
        r = np.linalg.norm(self.__agents[i] - self.__agents[j]) 
        # Calculamos el ATRACTIVO de la luciernga
        # beta = atraccion mutua / 
                # (1 + coeficiente de absorcion de la luz del medio * distancia luciernagas al cuadrado)
        beta = csi / (1 + psi * r ** 2) # atractivo de la luciernaga i sobre k


	# Calculamos la nueva posicion aplicando esta formula (distinta a la de las transparencias)
	# fi(t+1) = fj(t) + beta(rik) * (fi(t) - fj(t)) + alpha * exp(-t) * aleatorio
	# el aleatorio se calcula usando una distribucion normal Gaussiana, entre los limites propuestos y con la 
	# dimension dicha
        self.__agents[i] = self.__agents[j] + beta * (
            self.__agents[i] - self.__agents[j]) + alpha * exp(-t) * \
                                                   np.random.normal(norm0,
                                                                    norm1,
                                                                    dimension)
