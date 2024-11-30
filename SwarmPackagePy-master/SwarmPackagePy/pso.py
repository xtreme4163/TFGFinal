import os
import numpy as np
import copy
from . import intelligence
from . import misfunciones as fn


# Clase para el PSO (hereda de intelligence)
class pso(intelligence.sw):
    """
    Particle Swarm Optimization
    
    """

    # Constructor para el pso
    def __init__(self, n, funcion, lb, ub, dimension, iteraciones,numeroColores,pintor,vMin, vMax, w=0.5, c1=1,
                 c2=1, imagen=""):
        """
        :param n: numero de individuos (Particulas)
        :param funcion: funcion objetivo que se aplica en el algoritmo
        :param lb: limite inferior del espacio de busqueda
        :param ub: limite superior del espacio de busqueda
        :param dimension: dimension del espacio de solucion (r)
        :param iteraciones: numero de iteraciones
        :param numeroColores: numero de colores de la nueva imagen
        :param pintor: booleano que se usa para saber si pintamos imagen al final o no.
        :param vMin: velocidad mínima del individuo
        :param vMax: velocidad máxima del individuo
        :param w: parametro inercia
        :param c1: parametro cognitivo (f1)
        :param c2: parametro social (f2)
        :param imagen: ruta de la imagen a procesar por el algoritmo
       
        """
	

	# Empezamos a inicializar la poblacion de particulas con su velocidad y posicion
	# LLama al constructor de intelligence, así inicializa las
	# posiciones y la mejor posicion.
        super(pso, self).__init__()
	# generamos numeros aletaorios uniformemente distribuidos.
	# Generamos entre lb y ub (limite inferior y limite superior)
	# Generamos un total de n*dimension numeros
        self.__agents = np.random.uniform(lb, ub, (n,numeroColores, dimension))
    
        velocity = np.zeros((n,numeroColores, dimension)) #Llenamos el vector de velocidad con 0
        
        # Inicializar Pbest, es decir, inicialmente las mejores posiciones de las particulas son las
        # primeras halladas.
        Pbest = copy.deepcopy(self.__agents)
        #Calculamos el fitness actual y lo guardamos
        fitnessP = [funcion(x,numeroColores,imagen) for x in self.__agents]
        fitnessA = fitnessP # Lo igualamos al de la posicion ACTUAL
        
        # Iniciamos Gbest con el valor de la particula con menor fitness 

        Gbest=copy.deepcopy(Pbest[np.array([fitnessA]).argmin()])
        # hasta aqui hemos inicializado el PSO


	# Este bucle se repite hasta que nos salimos del rango iteraciones 
        for t in range(iteraciones):
           """
	   ESQUEMA PSO
	   1.1- evaluar fitness de cada particula
	   1.2 - actualizar mejor solucion personal de cada particula
	   1.3 - actualizar la mejor solucion global
	   
	   2 - Actualizar velocidad y posicion de cada particula
	   3 - mostrar resultado al acabar bucle 
	   """
           # Cálculo de la nueva velocidad
           velocity = self.calcularNuevaVelocidad(n, dimension, numeroColores, w, c1, c2, velocity, Pbest, Gbest,vMin,vMax)
           
           #Ajustamos la posicion de las particulas sumando la velocidad
           self.__agents += velocity
           # Ajusta esta posicion a los limites del espacio
           self.__agents = np.clip(self.__agents, lb, ub)

           #Se calcula el fitness actual de cada individuo
           fitnessA= [funcion(x,numeroColores,imagen) for x in self.__agents]
           #Actualizar mejor solucion particular
           #Para todas las particulas ...
           for i in range(n):
              # Si el fitness actual del individuo i es menor que su mejor fitness se actualiza
              if(fitnessA[i] < fitnessP[i]):
                 Pbest[i] = copy.deepcopy(self.__agents[i])
                 fitnessP[i] = fitnessA[i]
              
           # Actualizar mejor solucion global
           #Gbest pasa a ser la mejor solucion particular de aquel individuo que tenga un menor fitness
           Gbest=copy.deepcopy(Pbest[np.array([fitnessP]).argmin()])
           
           self.setMejorFitness(fitnessP[np.array([fitnessP]).argmin()])
           print(self.getMejorFitness(), end= ' ')

       ##########################################################################################################


        #Si se tiene que pintar la imagen generamos una cuantizada para su impresion.
        Gbest = np.int_(Gbest)
        # Generamos la imagen cuantizada para imprimirla con el mejor valor final global.
        reducida = fn.generaCuantizada(Gbest,numeroColores,imagen)

        #Pintamos imagen
        fn.pintaImagen(reducida, imagen,pintor,"PSO", numeroColores)

    def calcularNuevaVelocidad(self, n, dimension, numeroColores, w, c1, c2, velocity, Pbest, Gbest,vMin,vMax):
        r1 = np.random.rand(n,numeroColores,dimension)
        r2 = np.random.rand(n,numeroColores,dimension)
           
           #  Calculamos la nueva velocidad
        velocity = w * velocity + c1 * r1 * (
                Pbest - self.__agents) + c2 * r2 * (
                Gbest - self.__agents)
           # Ajustamos la velocidad para que no se salga de los limites. 
        velocity= np.clip(velocity, vMin, vMax)
        return velocity
        
        
