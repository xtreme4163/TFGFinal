import os
import numpy as np
from . import intelligence
from . import misfunciones as fn
import sys

# Limites para la velocidad de la particula
V_MAX = 4
V_MIN = -4

# numero de colores de la paleta
r = 64

# Clase para el PSO (hereda de intelligence)
class pso(intelligence.sw):
    """
    Particle Swarm Optimization
    
    """

    # Constructor para el pso
    def __init__(self, n, function, lb, ub, dimension, iteration,numeroColores,pintor, w=0.5, c1=1,
                 c2=1, imagen=""):
        """
        n: numero de individuos (Particulas)
        function: funcion que se aplica en el algoritmo
        lb: limite inferior del espacio de busqueda
        ub: limite superior del espacio de busqueda
        dimension: dimension del espacio de solucion (r)
        iteration: numero de iteraciones
        numeroColores: numero de colores de la nueva imagen
        pintor: booleano que se usa para saber si pintamos imagen al final o no.
        w: parametro inercia
        c1: parametro cognitivo (f1)
        c2: parametro social (f2)
        imagen: ruta de la imagen a procesar por el algoritmo
       
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
        Pbest = self.__agents
        #Calculamos el fitness actual y lo guardamos
        fitnessP = [function(x,numeroColores,imagen) for x in self.__agents]
        fitnessA = fitnessP # Lo igualamos al de la posicion ACTUAL
        
        # Iniciamos Gbest con el valor de la particula con menor fitness 

        Gbest=Pbest[np.array([fitnessA]).argmin()]
        # hasta aqui hemos inicializado el PSO


	# Este bucle se repite hasta que nos salimos del rango iteration 
	# Es el algoritmo PSO en sí
        for t in range(iteration):
           """
	   ESQUEMA PSO
	   1.1- evaluar fitness de cada particula
	   1.2 - actualizar mejor solucion personal de cada particula
	   1.3 - actualizar la mejor solucion global
	   
	   2 - Actualizar velocidad y posicion de cada particula
	   3 - mostrar resultado al acabar bucle 
	   """
           r1 = np.random.rand(n,numeroColores,dimension)
           r2 = np.random.rand(n,numeroColores,dimension)
           
           #  Calculamos la nueva velocidad
           velocity = w * velocity + c1 * r1 * (
                Pbest - self.__agents) + c2 * r2 * (
                Gbest - self.__agents)
           # Ajustamos la velocidad para que no se salga de los limites. 
           velocity= np.clip(velocity, V_MIN, V_MAX)
           
           #Ajustamos la posicion de las particulas sumando la velocidad
           self.__agents += velocity
           # Ajusta esta posicion a los limites del espacio
           self.__agents = np.clip(self.__agents, lb, ub)

           #Se calcula el fitness actual de cada individuo
           fitnessA= [function(x,numeroColores,imagen) for x in self.__agents]
           #Actualizar mejor solucion particular
           #Para todas las particulas ...
           for i in range(n):
              # Si el fitness actual del individuo i es menor que su mejor fitness se actualiza
              if(fitnessA[i] < fitnessP[i]):
                 Pbest[i] = self.__agents[i]
                 fitnessP[i] = fitnessA[i]
              
           # Actualizar mejor solucion global
           #Gbest pasa a ser la mejor solucion particular de aquel individuo que tenga un menor fitness
           Gbest=Pbest[np.array([fitnessP]).argmin()]
           
           self.setMejorFitness(fitnessP[np.array([fitnessP]).argmin()])
           print(self.getMejorFitness(), end= ' ')

       ##########################################################################################################

        if(pintor):
                #Si se tiene que pintar la imagen generamos una cuantizada para su impresion.
                Gbest = np.int_(Gbest)
                # Generamos la imagen cuantizada para imprimirla con el mejor valor final global.
                reducida = fn.generaCuantizada(Gbest,numeroColores,imagen)

                #Pintamos imagen
                fn.pintaImagen(reducida, imagen,pintor,"PSO", numeroColores)
        
        
