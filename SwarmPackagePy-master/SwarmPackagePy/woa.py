import os
import numpy as np
from . import intelligence
from . import misfunciones as fn

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


    def __init__(self, n, function, lb, ub, dimension, iteration,numeroColores,pintor, imagen=""):
        # Empezamos a inicializar la poblacion de individuos
        super(woa, self).__init__()

        # Inicializamos la población de ballenas
        self.__agents = np.random.uniform(lb, ub, (n, numeroColores, dimension))
        
        # Inicializar la mejor solución encontrada
        self.Gbest = self.__agents[0]
        self.best_fitness = float("inf")
        
        # Evaluar el fitness inicial de cada ballena
        self.fitness = [function(x, numeroColores, imagen) for x in self.__agents]
        for i in range(n):
            if self.fitness[i] < self.best_fitness:
                self.Gbest = self.__agents[i]
                self.best_fitness = self.fitness[i]
        
        # Parámetros del WOA
        a = 2  # Inicializamos a
        b = 1  # Parámetro para el movimiento en espiral
        
        # Iteraciones del algoritmo
        for t in range(iteration):
            a = 2 - t * (2 / iteration)  # Decrecimiento lineal de "a"
            
            # Para cada ballena
            for i in range(n):
                p = np.random.rand()  # Número aleatorio para decidir el comportamiento
                
                if p < 0.5:
                    A = 2 * a * np.random.rand() - a
                    C = 2 * np.random.rand()
                    if np.abs(A) < 1:
                        # Movimiento alrededor de la mejor solución (X*)
                        D = np.abs(C * self.Gbest - self.__agents[i])
                        self.__agents[i] = self.Gbest - A * D
                    else:
                        # Exploración: seleccionamos una ballena aleatoria
                        rand_ballena = self.__agents[np.random.randint(0, n)]
                        D = np.abs(C * rand_ballena - self.__agents[i])
                        self.__agents[i] = rand_ballena - A * D
                else:
                    # Movimiento en espiral
                    D = np.abs(self.Gbest - self.__agents[i])
                    l = np.random.uniform(-1, 1)
                    self.__agents[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.Gbest

                # Limitar las posiciones dentro del espacio de búsqueda
                self.__agents[i] = np.clip(self.__agents[i], lb, ub)

                # Calcular el fitness de la nueva posición
                new_fitness = function(self.__agents[i], numeroColores, imagen)
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.Gbest = self.__agents[i]
                        self.best_fitness = new_fitness

            # Mostrar el mejor fitness de la iteración
            self.setMejorFitness(self.best_fitness)
            print(self.getMejorFitness(), end=' ')

        # Guardamos la mejor solución encontrada y generamos la imagen cuantizada
        Gbest = np.int_(self.Gbest)
        reducida = fn.generaCuantizada(Gbest, numeroColores, imagen)
        fn.pintaImagen(reducida, imagen, pintor, "WOA", numeroColores)