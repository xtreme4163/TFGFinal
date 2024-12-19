import numpy as np
from . import intelligence
from . import misfunciones as fn
import copy

# Clase para abejas (hereda de intelligence)
class abejas(intelligence.sw):

        def __init__(self, n, funcion, lb, ub, dimension, iteraciones,numeroColores,pintor, imagen="", ajuste=0):
            """ 
                :param n: numero de individuos (Particulas)
                :param funcion: funcion objetivo que se aplica en el algoritmo
                :param lb: limite inferior del espacio de busqueda
                :param ub: limite superior del espacio de busqueda
                :param dimension: dimension del espacio de solucion
                :param iteraciones: numero de iteraciones
                :param numeroColores: numero de colores de la imagen cuantizada
                :param pintor: booleano que se usa para saber si pintamos las imagenes al final.
                :param imagen (str): Ruta a la imagen que debe procesarse.
                :param ajuste: parametro para decidir si se ajusta la paleta cuantizada a la imagen original       

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
            self.function = funcion  # función de fitness a optimizar
            self.lb = lb  # límite inferior del espacio de búsqueda
            self.ub = ub  # límite superior del espacio de búsqueda
            self.dimension = dimension  # número de dimensiones de cada solución
            self.iteration = iteraciones  # número de ciclos de optimización
            self.numeroColores = numeroColores  # colores para la cuantificación
            self.pintor = pintor  # bandera para pintar la imagen al final
            self.imagen = imagen  # ruta de la imagen a procesar
            self.ajuste=ajuste #ajuste

            
            # Generamos numeros aletaorios uniformemente distribuidos.
	        # Generamos entre lb y ub (limite inferior y limite superior)
	        # Generamos un total de n*dimension numeros
            self.__agents = np.random.uniform(lb, ub, (n,numeroColores, dimension))

            #Calculamos el fitness actual y lo guardamos
            self.fitnessActual = [funcion(x,numeroColores,imagen, ajuste) for x in self.__agents]
            fitnessMejor = self.fitnessActual # Lo igualamos al de la posicion ACTUAL
           

            #Se incializa también el "agotamiento" de las fuentes de alimento a 0
            limit = np.zeros(n)  
            self.limit = limit


            #Bucle del algoritmo
            for t in range(iteraciones):
                for i in range(n):
                    #Para cada fuente aplicamos las operaciones de las abejas empleadas
                    self.abejaEmpleada(i)
                self.abejasObservadoras()
                self.abejasExploradoras()

                #Actualizar mejor solucion particular
                #Para todas las fuentes  ...
                for i in range(n):
                    # Si el fitness actual del individuo i es menor que su mejor fitness se actualiza
                    if(self.fitnessActual[i] < fitnessMejor[i]):
                        fitnessMejor[i] = self.fitnessActual[i]

                # Actualizar mejor solucion global
                #Gbest pasa a ser la mejor solucion particular de aquel individuo que tenga un menor fitness
                Gbest=copy.deepcopy(self.__agents[np.array([fitnessMejor]).argmin()])
            
                self.setMejorFitness(fitnessMejor[np.array([fitnessMejor]).argmin()])
                print(self.getMejorFitness(), end= ' ')
            ##################################################################################################### Fin bucle

            # Generamos la imagen cuantizada para imprimirla con el mejor valor final global.
            reducida = fn.generaCuantizada(Gbest,numeroColores, imagen, ajuste)

            #print("Su fitness es: ", self.getMejorFitness())
            #Pintamos imagen
            fn.pintaImagen(reducida, imagen,pintor,"ABA",numeroColores)

        
        def abejaEmpleada(self, i):
            """
            Fase de abejas empleadas: Exploran alrededor de sus fuentes de alimento actuales buscando una mejor solución.
            """
            # Generar un nuevo candidato basado en la posición actual
            nuevaFuente = self.buscarFuenteVecina(i)
            
            # Evaluar la nueva solución
            fitNuevo = self.function(nuevaFuente, self.numeroColores, self.imagen, self.ajuste)
            
            # Si mejora la solución actual, la reemplazamos
            if fitNuevo < self.fitnessActual[i]:
                self.__agents[i] = copy.deepcopy(nuevaFuente)
                self.fitnessActual[i] = fitNuevo
                self.limit[i] = 0  # Resetear el contador de intentos fallidos
            else:
                #Si no se mejora la solucion se incrementa el "agotamiento" de la fuente
                self.limit[i] += 1

        def abejasObservadoras(self):
            """
            Fase de abejas observadoras: Seleccionan soluciones basadas en su calidad y exploran alrededor de ellas.
            """
            fitness_total = sum(self.fitnessActual)

            #Se calculan las probabilidades de todas las fuentes
            probabilidades = [fit / fitness_total for fit in self.fitnessActual]
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
            abandono = self.iteration / 0.2  # número de intentos fallidos antes de abandonar una fuente. Proporcionar al numero de iteraciones            
            for i in range(self.n):
                if self.limit[i] > abandono:
                    # Reemplazar con una nueva solución aleatoria
                    self.__agents[i] = copy.deepcopy(np.random.uniform(self.lb,self.ub,(self.numeroColores, self.dimension)))
                    #Se calcula su fitnes actual y se resetea el contador de "agotamiento"
                    self.fitnessActual[i] = self.function(self.__agents[i], self.numeroColores, self.imagen,self.ajuste)
                    self.limit[i] = 0  # Resetear el contador de intentos
                    
