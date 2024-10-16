import numpy as np
from . import intelligence
from . import misfunciones as fn

# Clase para abejas (hereda de intelligence)
class abejas(intelligence.sw):

        def __init__(self, n, function, lb, ub, dimension, iteration,numeroColores,pintor, imagen=""):
            """ 
                n (int): Número de abejas (fuentes de alimento) en la población.
                function: Función objetivo que queremos optimizar.
                lb (list): Límite inferior del espacio de búsqueda para cada dimensión.
                ub (list): Límite superior del espacio de búsqueda para cada dimensión.
                dimension (int): Dimensiones del espacio de búsqueda.
                iteration (int): Número de iteraciones que debe ejecutar el algoritmo.
                numeroColores (int): Número de colores en la imagen.
                pintor (bool): Control booleano para pintar o no la imagen al final.
                imagen (str): Ruta a la imagen que debe procesarse.

                PSEUDOCODIGO
                Inicializar el conjunto de fuentes de alimento
                Repetir
                    Operaciones de abejas empleadas
                    Operaciones de abejas observadoras
                    Operaciones de abejas exploradoras
                    Actualizar la mejor solución hasta el momento
                hasta que (condición de parada)

            
            """
                
            # Empezamos a inicializar la poblacion de individuos
            # LLama al constructor de intelligence, así inicializa las
            # posiciones y la mejor posicion.
            super(abejas, self).__init__()

            # Parámetros básicos
            self.n = n  # número de fuentes de alimento (abejas)
            self.function = function  # función de fitness a optimizar
            self.lb = lb  # límite inferior del espacio de búsqueda
            self.ub = ub  # límite superior del espacio de búsqueda
            self.dimension = dimension  # número de dimensiones de cada solución
            self.iteration = iteration  # número de ciclos de optimización
            self.numeroColores = numeroColores  # colores para la cuantificación
            self.pintor = pintor  # bandera para pintar la imagen al final
            self.imagen = imagen  # ruta de la imagen a procesar

            #Se inicializan 
            #self.__agents =  np.random.uniform(lb, ub, (n,numeroColores, dimension))
            self.__agents = np.zeros((n, numeroColores, dimension))

            for i in range(n):  # Para cada fuente
                for j in range(numeroColores):  # Para cada color
                    for d in range(dimension):  # Para cada componente (R, G, B, etc.)
                        # Inicializa la componente j de la fuente i
                        self.__agents[i, j, d] = lb + np.random.uniform(0, 1) * (ub - lb)

            # Inicializar Pbest, es decir, inicialmente las mejores posiciones de las particulas son las
            # primeras halladas.
            Pbest = self.__agents
            self.Pbest = Pbest
            #Calculamos el fitness actual y lo guardamos
            fitActual = [function(x,numeroColores,imagen) for x in self.__agents]
            fitMejor = fitActual # Lo igualamos al de la posicion ACTUAL
            self.fitActual = fitActual

            #Se incializa también el "agotamiento" de las fuentes de alimento a 0
            limit = np.zeros(n)  
            self.limit = limit


            #Bucle del algoritmo
            for t in range(iteration):
                for i in range(n):
                    #Para cada fuente aplicamos las operaciones de las abejas empleadas
                    self.abejaEmpleada(i)
                self.abejasObservadoras()
                self.abejasExploradoras()

                #Actualizar mejor solucion particular
                #Para todas las fuentes  ...
                for i in range(n):
                    # Si el fitness actual del individuo i es menor que su mejor fitness se actualiza
                    if(self.fitActual[i] < fitMejor[i]):
                        Pbest[i] = self.__agents[i]
                        fitMejor[i] = self.fitActual[i]

                # Actualizar mejor solucion global
                #Gbest pasa a ser la mejor solucion particular de aquel individuo que tenga un menor fitness
                Gbest=Pbest[np.array([fitMejor]).argmin()]
            
                self.setMejorFitness(fitMejor[np.array([fitMejor]).argmin()])
                print(self.getMejorFitness(), end= ' ')
            ##################################################################################################### Fin bucle

            # Generamos la imagen cuantizada para imprimirla con el mejor valor final global.
            reducida = fn.generaCuantizada(Gbest,numeroColores, imagen)

            #print("Su fitness es: ", self.getMejorFitness())
            #Pintamos imagen
            fn.pintaImagen(reducida, imagen,pintor,"ABA",numeroColores)


        def buildFuentes(self):
            """
            Inicializa las fuentes de alimento (soluciones) utilizando la fórmula
            x_ij = x_j_min + γ * (x_j_max - x_j_min)
            """
            # Número aleatorio γ en [0, 1]
            gamma = np.random.rand(self.n, self.dimension)

            # Inicialización utilizando la fórmula: lb + γ * (ub - lb)
            sources = self.lb + gamma * (self.ub - self.lb)

            return sources
        

        def abejaEmpleada(self, i):
            """
            Fase de abejas empleadas: Exploran alrededor de sus fuentes de alimento actuales buscando una mejor solución.
            """
            # Generar un nuevo candidato basado en la posición actual
            nuevaFuente = self.buscarFuenteVecina(i)
            
            # Evaluar la nueva solución
            fitNuevo = self.function(nuevaFuente, self.numeroColores, self.imagen)
            
            # Si mejora la solución actual, la reemplazamos
            if fitNuevo < self.fitActual[i]:
                self.__agents[i] = nuevaFuente
                self.fitActual[i] = fitNuevo
                self.limit[i] = 0  # Resetear el contador de intentos fallidos
            else:
                #Si no se mejora la solucion se incrementa el "agotamiento" de la fuente
                self.limit[i] += 1

        def abejasObservadoras(self):
            """
            Fase de abejas observadoras: Seleccionan soluciones basadas en su calidad y exploran alrededor de ellas.
            """
            fitness_total = sum(self.fitActual)

            #Se calculan las probabilidades de todas las fuentes
            probabilidades = [fit / fitness_total for fit in self.fitActual]
            #Se coge el índice de la fuente con mayor probabilidad
            fuenteSeleccionada = np.argmax(probabilidades)
            
            #Operamos sobre esa fuente como una abeja empelada
            self.abejaEmpleada(fuenteSeleccionada)
           

        
        def buscarFuenteVecina(self, i):
            """
            Genera una nueva solución (fuente de alimento) alrededor de la solución actual (i).
            """
            #Se genera un numero aleatorio entre -1 y 1 con las dimensiones correctas
            aleatorio = np.random.uniform(-1, 1, size=self.dimension)
            # Se selecciona una fuente al azar
            while True:
                j = np.random.randint(0,self.n)
                if(j != i):
                    #Se para de buscar una fuente vecina cuando no sea la misma que la que se estaba evaluando.
                    break      

            #Se encuentra la fuente candidata siguiendo la fórmula.
            nuevaFuente = self.__agents[i] + aleatorio * (self.__agents[i] - self.__agents[j])
            #Se ajusta a los limites permitidos
            return np.clip(nuevaFuente, self.lb, self.ub)  
        

        def abejasExploradoras(self):
            """
            Fase de abejas exploradoras: Abandonan las fuentes de alimento que no mejoran durante un número determinado de ciclos y buscan nuevas soluciones aleatorias.
            """
            abandono = 15  # número de intentos fallidos antes de abandonar una fuente
            for i in range(self.n):
                if self.limit[i] > abandono:
                    # Reemplazar con una nueva solución aleatoria
                    for j in range(self.numeroColores):  # Para cada color
                        for d in range(self.dimension):  # Para cada componente (R, G, B, etc.)
                            # Inicializa la componente j de la fuente i
                            self.__agents[i, j, d] = self.lb + np.random.uniform(0, 1) * (self.ub - self.lb)
                    #Se calcula su fitnes actual y se resetea el contador de "agotamiento"
                    self.fitActual[i] = self.function(self.__agents[i], self.numeroColores, self.imagen)
                    self.limit[i] = 0  # Resetear el contador de intentos
                    
