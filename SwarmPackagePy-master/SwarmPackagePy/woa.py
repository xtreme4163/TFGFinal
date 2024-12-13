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


    def __init__(self, n, funcion, lb, ub, dimension, iteraciones,numeroColores,pintor, imagen=""):

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

        """
         
        # Empezamos a inicializar la poblacion de individuos
        super(woa, self).__init__()

        # Inicializamos la población de ballenas
        self.__agents = np.random.uniform(lb, ub, (n, numeroColores, dimension))
        
        # Inicializar la mejor solución encontrada
        self.Gbest = copy.deepcopy(self.__agents[0]) 
        self.mejorFitness = float("inf")
        
        # Evaluar el fitness inicial de cada ballena
        self.fitness = [funcion(x, numeroColores, imagen) for x in self.__agents]
        # cambiar esto self.Gbest=copy.deepcopy(Pbest[np.array([fitnessActual]).argmin()])

        for i in range(n):
            if self.fitness[i] < self.mejorFitness:
                self.Gbest = self.__agents[i]
                self.mejorFitness = self.fitness[i]
        
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
                        D = np.abs(C * (self.Gbest - self.__agents[i]))
                        new_agents[i] = self.Gbest - A * D
                    else:
                        # Exploración: seleccionamos una ballena aleatoria
                        ind = np.random.randint(0, n-1)
                        while(ind == i)
                            ind = np.random.randint(0, n-1)
                        ballenaAleatoria = self.__agents[ind]
                        D = np.abs(C * (ballenaAleatoria - self.__agents[i]))
                        new_agents[i] = ballenaAleatoria - A * D
                else:
                    # Movimiento en espiral
                    l = np.random.uniform(-1, 1)
                    D = np.abs(C * (self.Gbest - self.__agents[i]))
                    new_agents[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.Gbest

                # Limitar las posiciones dentro del espacio de búsqueda
                new_agents[i] = np.clip(new_agents[i], lb, ub)

            
            self.__agents = copy.deepcopy(new_agents)
            # Calcular el fitness de la nueva posición
            fitnessNuevo = funcion(self.__agents[i], numeroColores, imagen)
            #Usar lo mismo del pso, para coger el mejor aqui la i ya no tiene sentido
            if fitnessNuevo < self.fitness[i]:
                self.fitness[i] = fitnessNuevo
                if fitnessNuevo < self.mejorFitness:
                    self.Gbest = copy.deepcopy(self.__agents[i])
                    self.mejorFitness = fitnessNuevo

            # Mostrar el mejor fitness de la iteración
            self.setMejorFitness(self.mejorFitness)
            print(self.getMejorFitness(), end=' ')

        # Guardamos la mejor solución encontrada y generamos la imagen cuantizada
        Gbest = np.int_(self.Gbest)
        reducida = fn.generaCuantizada(Gbest, numeroColores, imagen)
        fn.pintaImagen(reducida, imagen, pintor, "WOA", numeroColores)