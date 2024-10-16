import numpy as np

from . import intelligence

from . import misfunciones as fn


import sys

# Limites para la velocidad de la particula
V_MAX = 4
V_MIN = -4

# numero de colores de la paleta
r = 8

# usar k-means del paquete sklearn.cluster
# Clase para el PSO (hereda de intelligence)
class pso(intelligence.sw):
    """
    Particle Swarm Optimization
    
    """


    """Para llamar al pso desde example.py
    alh = SwarmPackagePy.pso(50, tf.easom_function, -10, 10, 3, 20,
                    w=0.5, c1=1, c2=1)"""
    # Constructor para el pso
    def __init__(self, n, function, lb, ub, dimension, iteration, w=0.5, c1=1,
                 c2=1):
        """
        :param n: number of agents / PARTICULAS
        :param function: test function / Funcion a aplicar por el PSO
        :param lb: lower limits for plot axes  / LIMITE INFERERIOR
        :param ub: upper limits for plot azes  / LIMITE SUPERIOR
        :param dimension: space dimension   / Dimension del espacio de solucion (r)
        :param iteration: the number of iterations / nº de iteraciones
        :param w: balance between the range of research and consideration for
        suboptimal decisions found (default value is 0.5): / inercia del PSO
        w>1 the particle velocity increases, they fly apart and inspect
         the space more carefully;
        w<1 particle velocity decreases, convergence speed depends
        on parameters c1 and c2 ;
        :param c1: ratio between "cognitive" and "social" component
        (default value is 1) Parámetro cognitivo (f1)
        :param c2: ratio between "cognitive" and "social" component
        (default value is 1) / Parámetro social (f2)
        """
	
	# Empezamos a inicializar la poblacion de particulas con su velocidad y posicion
	# LLama al constructor de intelligence, así inicializa las
	# posiciones y la mejor posicion.
        super(pso, self).__init__()


	# generamos numeros aletaorios uniformemente distribuidos.
	# Generamos entre lb y ub (limite inferior y limite superior)
	# Generamos un total de n*dimension numeros
        #self.__agents = np.random.randint(lb, ub, (n,r, dimension))
        self.__agents = np.random.uniform(lb, ub, (n,r, dimension))
    
        velocity = np.zeros((n,r, dimension)) #Llenamos el vector de velocidad con 0
        self._points(self.__agents)
        #print(self.__agents)
        
        # Inicializar Pbest, es decir, inicialmente las mejores posiciones de las particulas son las
        # primeras halladas.
        Pbest = self.__agents
       
        
        # Iniciamos Gbest con el valor de la particula con menor fitness 
        Gbest = Pbest[np.array([function(x,r) for x in Pbest]).argmin()] 
        #Gbest = Pbest[np.array([function(x,r) for x in Pbest]).argmin()]
        #print("Esto es Gbest inicial")
        #print(Gbest)
        # hasta aqui hemos inicializado el PSO


        print("PSO // Particulas: ",n, "Colores: ",r,"Iteraciones: ", iteration)
	# Este bucle se repite hasta que nos salimos del rango iteration 
	# Es el algoritmo PSO en sí
        for t in range(iteration):
           print("Iteraccion ", t+1)
           """
	   ESQUEMA PSO
	   1.1- evaluar fitness de cada particula
	   1.2 - actualizar mejor solucion personal de cada particula
	   1.3 - actualizar la mejor solucion global
	   
	   2 - Actualizar velocidad y posicion de cada particula
	   3 - mostrar resultado al acabar bucle 
	   """
           r1 = np.random.rand(n,r,dimension)
           r2 = np.random.rand(n,r,dimension)
           
           #  Calculamos la nueva velocidad
           velocity = w * velocity + c1 * r1 * (
                Pbest - self.__agents) + c2 * r2 * (
                Gbest - self.__agents)
           # Ajustamos la velocidad para que no se salga de los limites. 
           velocity= np.clip(velocity, V_MIN, V_MAX)
           #print(velocity)
           
           #Ajustamos la posicion de las particulas sumando la velocidad
           self.__agents += velocity
           # Ajusta esta posicion a los limites del espacio
           self.__agents = np.clip(self.__agents, lb, ub)
           #Y lo convertimos en lista
            
           self._points(self.__agents)

           #Actualizar mejor solucion particular
           #Para todas las particulas ...
           for i in range(n):
              # Si el fitness de esta posicion es menor que el almacenado lo actualizamos
              if(function(self.__agents[i],r) < function(Pbest[i],r)):
                 Pbest[i] = self.__agents[i]
              
           # Actualizar mejor solucion global
           Gbest = Pbest[np.array([function(x,r) for x in Pbest]).argmin()] 
           
         
           print("Fitness --> ", function(Gbest,r) )
        #Guardamos la mejor solucion encontrada por el algoritmo
        Gbest = np.int_(Gbest)
        self._set_Gbest(Gbest)
        # Generamos la imagen cuantizada para imprimirla con el mejor valor final global.
        reducida = fn.genera_cuantizada(Gbest,r)
        # Esto pinta la mejor paleta
        #print(Gbest)
        
        #Pintamos imagen
        fn.pintaImagen(reducida)
        print("Su fitness es: ", function(Gbest,r))
