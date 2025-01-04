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
        
        # Evaluar el fitness actual de cada ballena
        fitnessActual = [funcion(x, numeroColores, imagen, ajuste) for x in self.__agents]
        #Inicialmente el fitnees de la mejor posicion personal de cada individuo es igual al fitness de su posicion actual
        fitnessMejor = fitnessActual

        #Inicializar la mejor solución encontrada
        indice_mejor = np.array([fitnessMejor]).argmin()
        Gbest=copy.deepcopy(self.__agents[indice_mejor])
        self.setMejorFitness(fitnessMejor[indice_mejor])  
        
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
                        D = np.abs(C * Gbest - self.__agents[i])
                        new_agents[i] = Gbest - A * D
                    else:
                        # Exploración: seleccionamos una ballena aleatoria
                        ballenaAleatoria = np.random.randint(0, n-1)
                        while(ballenaAleatoria == i):
                            ballenaAleatoria = np.random.randint(0, n-1)
                        D = np.abs(C * ( self.__agents[ballenaAleatoria] - self.__agents[i]))
                        new_agents[i] = self.__agents[ballenaAleatoria] - A * D
                else:
                    # Movimiento en espiral
                    l = np.random.uniform(-1, 1)
                    D = np.abs(Gbest - self.__agents[i])
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
                    fitnessMejor[i] = fitnessActual[i]


            #Actualizar mejor solucion personal
            indice_mejor = np.array([fitnessMejor]).argmin()
            if fitnessMejor[indice_mejor] < self.getMejorFitness():
                Gbest=copy.deepcopy(self.__agents[indice_mejor])
                self.setMejorFitness(fitnessMejor[indice_mejor])    

            # Mostrar el mejor fitness de la iteración
            self.setMejorFitness(fitnessMejor[np.array([fitnessMejor]).argmin()])
            print(self.getMejorFitness(), end=' ')

        # Guardamos la mejor solución encontrada y generamos la imagen cuantizada
        Gbest = np.int_(Gbest)
        reducida = fn.generaCuantizada(Gbest, imagen, ajuste)
        fn.pintaImagen(reducida, imagen, pintor, "WOA", numeroColores)