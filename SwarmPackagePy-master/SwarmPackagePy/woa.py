import os
import numpy as np
from . import intelligence
from . import misfunciones as fn
import copy

class woa(intelligence.sw):

    """
    1. Inicializar la población de ballenas (posiciones) de forma aleatoria.
    2. Evaluar el fitness de cada ballena.
    3. Identificar la mejor solución global (X*).
    4. Mientras no se alcance el criterio de parada:
    5. Para cada ballena (i):
      6. Generar un número aleatorio `p` entre 0 y 1.
      7. Si `p < 0.5`:
         8. Si `|A| < 1`, actualizar la posición con respecto a la mejor solución (X*).
         9. Si `|A| >= 1`, seleccionar una ballena aleatoria y actualizar la posición.
      10. Si `p >= 0.5`, mover la ballena en un camino en espiral alrededor de la mejor solución.
    11. Actualizar la mejor solución global si se encuentra una mejor."""


    def __init__(self, n, funcion, lb, ub, dimension, iteraciones,numeroColores,pintor, imagen="", ajuste=0):

        """     
        :param n: numero de individuos
        :param funcion: funcion objetivo que se aplica en el algoritmo
        :param lb: limite inferior del espacio de busqueda
        :param ub: limite superior del espacio de busqueda
        :param dimension: dimension del espacio de solucion
        :param iteraciones: numero de iteraciones
        :param numeroColores: numero de colores de la nueva imagen
        :param pintor: booleano que se usa para saber si pintamos las imagenes al final.
        :param imagen: ruta de la imagen a procesar por el algoritmo
        :param ajuste: parametro para decidir si se ajusta la paleta cuantizada a la imagen original       

        """
         
        # Empezamos a inicializar la poblacion de individuos
        super(woa, self).__init__()

        # Inicializamos la población de ballenas
        self.__agents = np.random.uniform(lb, ub, (n, numeroColores, dimension))
        
        # Inicializar Pbest, es decir, inicialmente las mejores posiciones de los individuos son las
        # primeras halladas.
        Pbest = copy.deepcopy(self.__agents)
        # Evaluar el fitness actual de cada ballena
        fitnessActual = [funcion(x, numeroColores, imagen, ajuste) for x in self.__agents]
        #Inicialmente el fitnees de la mejor posicion personal de cada individuo es igual al fitness de su posicion actual
        fitnessMejor = fitnessActual

        #Inicializar la mejor solución encontrada
        Gbest = copy.deepcopy(Pbest[np.array([fitnessActual]).argmin()])
        
        # Parámetros del WOA
        a = 2  # Inicializamos a
        b = 1  # Parámetro para el movimiento en espiral
        
        # Iteraciones del algoritmo
        for t in range(iteraciones):
            a = 2 - 2 * (t / iteraciones)
            A = 2 * a * np.random.rand(numeroColores, dimension) - a
            C = 2 * np.random.rand(numeroColores, dimension)
            

            new_agents = copy.deepcopy(self.__agents)
            for i in range(n):
                p = np.random.rand()
                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        # Movimiento alrededor de la mejor solución (X*)
                        D = np.abs(C * (Gbest - self.__agents[i]))
                        new_agents[i] = Gbest - A * D
                    else:
                        # Exploración: seleccionamos una ballena aleatoria
                        ind = np.random.randint(0, n-1)
                        while(ind == i):
                            ind = np.random.randint(0, n-1)
                        ballenaAleatoria = self.__agents[ind]
                        D = np.abs(C * (ballenaAleatoria - self.__agents[i]))
                        new_agents[i] = ballenaAleatoria - A * D
                else:
                    # Movimiento en espiral
                    l = np.random.uniform(-1, 1)
                    D = np.abs(C * (Gbest - self.__agents[i]))
                    new_agents[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + Gbest

                # Limitar las posiciones dentro del espacio de búsqueda
                new_agents[i] = np.clip(new_agents[i], lb, ub)

            
            self.__agents = copy.deepcopy(new_agents)
            #Se calcula el fitness de la posicion actual de cada individuo
            fitnessActual= [funcion(x,numeroColores,imagen, ajuste) for x in self.__agents]
            #Actualizar mejor solucion personal
            #Para todas los individuos ...
            for i in range(n):
              # Si el fitness de la posicion actual del individuo i es menor que el fitness de su posicion personal se actualiza
              if(fitnessActual[i] < fitnessMejor[i]):
                 Pbest[i] = copy.deepcopy(self.__agents[i])
                 fitnessMejor[i] = fitnessActual[i]

            # Actualizar mejor solucion global
            #Gbest pasa a ser la mejor solucion particular de aquel individuo que tenga un menor fitness
            Gbest=copy.deepcopy(Pbest[np.array([fitnessMejor]).argmin()])

            # Mostrar el mejor fitness de la iteración
            self.setMejorFitness(fitnessMejor[np.array([fitnessMejor]).argmin()])
            print(self.getMejorFitness(), end=' ')

        # Guardamos la mejor solución encontrada y generamos la imagen cuantizada
        Gbest = np.int_(Gbest)
        reducida = fn.generaCuantizada(Gbest, numeroColores, imagen, ajuste)
        fn.pintaImagen(reducida, imagen, pintor, "WOA", numeroColores)