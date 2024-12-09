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
    def __init__(self, n, funcion, lb, ub, dimension, iteraciones,numeroColores,pintor,vMin, vMax, w=0.5, f1=1,
                 f2=1, imagen=""):
        """
        :param n: numero de individuos (Particulas)
        :param funcion: funcion objetivo que se aplica en el algoritmo
        :param lb: limite inferior del espacio de busqueda
        :param ub: limite superior del espacio de busqueda
        :param dimension: dimension del espacio de solucion
        :param iteraciones: numero de iteraciones
        :param numeroColores: numero de colores de la imagen cuantizada
        :param pintor: booleano que se usa para saber si pintamos las imagenes al final.
        :param vMin: velocidad mínima del individuo
        :param vMax: velocidad máxima del individuo
        :param w: parametro inercia
        :param f1: parametro cognitivo
        :param f2: parametro social
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
    
        #Llenamos el vector de velocidad de cada individuo con 0
        velocity = np.zeros((n,numeroColores, dimension)) 
        
        # Inicializar Pbest, es decir, inicialmente las mejores posiciones de las particulas son las
        # primeras halladas.
        Pbest = copy.deepcopy(self.__agents)
        #Calculamos el fitness de la mejor posicion actual de todos los individuos y lo guardamos
        fitnessActual = [funcion(x,numeroColores,imagen) for x in self.__agents]
        #Inicialmente el fitnees de la mejor posicion personal de cada individuo es igual al fitness de su posicion actual
        fitnessMejor = fitnessActual 
        
        # Iniciamos Gbest con la posicion de la particula con menor fitness 

        Gbest=copy.deepcopy(Pbest[np.array([fitnessActual]).argmin()])
        # hasta aqui hemos inicializado el PSO


	# Este bucle se repite hasta que nos salimos del rango iteraciones 
        for t in range(iteraciones):
           """
	   ESQUEMA PSO
                 - Actualizar velocidad y posicion de cada particula

	  - evaluar fitness de cada particula
	    - actualizar mejor solucion personal de cada particula
	    - actualizar la mejor solucion global
	   
	   """
           # Cálculo de la nueva velocidad
           velocity = calcularNuevaVelocidad(self.__agents, n, dimension, numeroColores, w, f1, f2, velocity, Pbest, Gbest,vMin,vMax)
           
           #Ajustamos la posicion de las particulas sumando la velocidad
           for i in range(n):
                #Se suma la velocidad nueva de i al individuo i
                self.__agents[i] += velocity[i]
                # Ajusta esta posicion a los limites del espacio
                self.__agents[i] = np.clip(self.__agents[i], lb, ub)


           #Se calcula el fitness de la posicion actual de cada individuo
           fitnessActual= [funcion(x,numeroColores,imagen) for x in self.__agents]
           #Actualizar mejor solucion personal
           #Para todas las particulas ...
           for i in range(n):
              # Si el fitness de la posicion actual del individuo i es menor que el fitness de su posicion personal se actualiza
              if(fitnessActual[i] < fitnessMejor[i]):
                 Pbest[i] = copy.deepcopy(self.__agents[i])
                 fitnessMejor[i] = fitnessActual[i]
              
           # Actualizar mejor solucion global
           #Gbest pasa a ser la mejor solucion particular de aquel individuo que tenga un menor fitness
           Gbest=copy.deepcopy(Pbest[np.array([fitnessMejor]).argmin()])
           
           self.setMejorFitness(fitnessMejor[np.array([fitnessMejor]).argmin()])
           print(self.getMejorFitness(), end= ' ')

       ##########################################################################################################


        #Si se tiene que pintar la imagen generamos una cuantizada para su impresion.
        Gbest = np.int_(Gbest)
        # Generamos la imagen cuantizada para imprimirla con el mejor valor final global.
        reducida = fn.generaCuantizada(Gbest,numeroColores,imagen)

        #Pintamos imagen
        fn.pintaImagen(reducida, imagen,pintor,"PSO", numeroColores)

""" 
Funcion que calcula la nueva velocidad del individuo aplicando la fórmula:
        w * velocidad + f1 * r1 * (Pbest(i) - x(i)) + f2 * r2 * (Gbest(i) - x(i))
Retorna la nueva velocidad de cada particula en un array.
"""
def calcularNuevaVelocidad(agents, n, dimension, numeroColores, w, f1, f2, velocidad, Pbest, Gbest,vMin,vMax):

        r1 = np.random.rand(n,numeroColores,dimension)
        r2 = np.random.rand(n,numeroColores,dimension)
        #Se incializa la nueva velocidad
        nuevaVelocidad = np.zeros_like(velocidad) 
                
        for i in range (n):
                #Calculo de la velocidad para cada individuo
                nuevaVelocidad[i] = w * velocidad[i] + f1 * r1[i] * (
                        Pbest[i] - agents[i]) + f2 * r2[i] * (
                        Gbest - agents[i])
                # Se ajusta la velocidad para que no se salga de los limites. 
                nuevaVelocidad[i] = np.clip(nuevaVelocidad[i], vMin, vMax)

        return nuevaVelocidad
        
        
