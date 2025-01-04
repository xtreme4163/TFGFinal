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
            self.__agents = self.buildFuentes()

            #Calculamos el fitness actual y lo guardamos
            self.fitnessActual = [funcion(x,numeroColores,imagen, ajuste) for x in self.__agents]
           
           # Iniciamos Gbest con la posicion de la fuente con menor fitness 
            indice_mejor = np.array([self.fitnessActual]).argmin()
            Gbest=copy.deepcopy(self.__agents[indice_mejor] )
            self.setMejorFitness(self.fitnessActual[indice_mejor] )

            #Se incializa también el "agotamiento" de las fuentes de alimento a 0
            limit = np.zeros(n)  
            self.limit = limit


            #Bucle del algoritmo
            for t in range(iteraciones):
                #Duplicado de la posicion actual.
                posActual = copy.deepcopy(self.__agents)

                for i in range(n):
                    #Para cada fuente aplicamos las operaciones de las abejas empleadas
                    self.abejaEmpleada(i, posActual)

                #Se repite la copia antes de aplicar las abejas observadoras
                posActual = copy.deepcopy(self.__agents)
                self.abejasObservadoras(posActual)
                self.abejasExploradoras()


                
                #Se calcula el fitness actual de cada fuente
                self.fitnessActual= [funcion(x,numeroColores,imagen, ajuste) for x in self.__agents]

                #Se actualiza la mejor solucion
                indice_mejor = np.array([self.fitnessActual]).argmin()
                if(self.fitnessActual[indice_mejor] < self.getMejorFitness()):
                    self.setMejorFitness(self.fitnessActual[indice_mejor])
                    Gbest = copy.deepcopy(self.__agents[indice_mejor])
      
                print(self.getMejorFitness(), end= ' ')
            ##################################################################################################### Fin bucle

            # Generamos la imagen cuantizada para imprimirla con el mejor valor final global.
            reducida = fn.generaCuantizada(Gbest,numeroColores, imagen, ajuste)

            #print("Su fitness es: ", self.getMejorFitness())
            #Pintamos imagen
            fn.pintaImagen(reducida, imagen,pintor,"ABA",numeroColores)

        
        def abejaEmpleada(self, i, posActual):
            """
            Fase de abejas empleadas: Exploran alrededor de sus fuentes de alimento actuales buscando una mejor solución.
            """
            # Generar un nuevo candidato basado en la posición actual
            nuevaFuente = self.buscarFuenteVecina(i, posActual)
            
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

        def abejasObservadoras(self, posic_actual):
            """
            Fase de abejas observadoras: Seleccionan soluciones basadas en su calidad y exploran alrededor de ellas.
            """
            # se determina la probabilidad asociada a cada flor. Para ello, se
            # normalizan los fitness (dividiéndolos entre su suma)
            fitness_total = sum(self.fitnessActual)         
            probabilidades = [fit / fitness_total for fit in self.fitnessActual]

            # se determina el número de flores
            sumandos = len(probabilidades)
            
            #Al haber normalizado los valores, no es necesario sumarlos otra vez. Puedo
            # considerar directamente el valor 1 como suma

            # se crea una lista de tuplas con dos elementos: (indice_flor, probabilidad)                                               
            parejas = [(x, probabilidades[x]) for x in range(0, sumandos)]            

            # se ordena la lista por el segundo elemento de las tuplas (es decir,
            # por la probabilidad)            
            parejas.sort(key=lambda x: x[1])       
             
            for i in range(self.n):
                fuenteSeleccionada = np.empty(sumandos, int)
                for i in range(0, sumandos):             
                    # se elige una flor/fuente aleatoriamente                        
                    fuenteSeleccionada[i] = self.seleccionRuleta(parejas, sumandos, 1) 
                                        
                for i in range(0, sumandos):             
                    #Operamos sobre esa fuente como una abeja empelada
                    self.abejaEmpleada(fuenteSeleccionada[i], posic_actual)
        
        def buscarFuenteVecina(self, i, posActual):
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
            nuevaFuente = posActual[i] + aleatorio * (posActual[i] - posActual[j])
            
            #Se ajusta a los limites permitidos
            return np.clip(nuevaFuente, self.lb, self.ub) 
        

        def abejasExploradoras(self):
            """
            Fase de abejas exploradoras: Abandonan las fuentes de alimento que no mejoran durante un número determinado de ciclos y buscan nuevas soluciones aleatorias.
            """
            abandono = self.iteration * 0.2  # número de intentos fallidos antes de abandonar una fuente. Proporcional al numero de iteraciones            
            for i in range(self.n):
                if self.limit[i] > abandono:
                    # Reemplazar con una nueva solución aleatoria
                    self.__agents[i] = copy.deepcopy(np.random.uniform(self.lb,self.ub,(self.numeroColores, self.dimension)))
                    #Se calcula su fitnes actual y se resetea el contador de "agotamiento"
                    self.fitnessActual[i] = self.function(self.__agents[i], self.numeroColores, self.imagen,self.ajuste)
                    self.limit[i] = 0  # Resetear el contador de intentos
                    
        
        def buildFuentes(self):
            """
            Inicializa las fuentes de alimento (soluciones) utilizando la fórmula
            x_ij = x_j_min + γ * (x_j_max - x_j_min)
            """
            # Número aleatorio γ en [0, 1]
            gamma = np.random.rand(self.n,self.numeroColores, self.dimension)
            # Inicialización utilizando la fórmula: lb + γ * (ub - lb)
            sources = self.lb + gamma * (self.ub - self.lb)

            return sources
        

         # ###################################################
        # Selección de una flor del conjunto actual mediante el método wheel-roulette
        #
        # PARÁMETROS:
        #  parejas: lista de tuplas de la forma (indice_flor, fitness)
        #  num_oparejas: número de tuplas de la lista 
        #  sum_fit: suma de los fitness de todas las tuplas (que se normalizaron previamente)
        # RETORNO:
        #   índice de la flor seleccionada  (entero en [0, num_parejas] )
        # ###################################################
        def seleccionRuleta(self, parejas, num_parejas, suma_fit):
           # Generar un número aleatorio entre 0 y la suma de fitness normalizados
           aleatorio = np.random.uniform(0, suma_fit)        
        
           acumulado = 0
           
           # se recorren las tuplas de la lista
           for i in range(0, num_parejas): 
              # se acumula el fitness de la tupla i-esima
              acumulado += parejas[i][1] 
              
              #si el acumulado supera el valor aleatorio, se toma la fuente i-esima
              if acumulado > aleatorio:
                 return parejas[i][0] 
