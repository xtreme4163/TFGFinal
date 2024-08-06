"""
 Clase para definir los métodos usados por todos los algoritmos, 
 como actualizar la g (mejor posicion) ...

"""
class sw(object):

    # Define las posiciones y las mejores posiciones
    def __init__(self):
        self.__Positions = []  #Se inicializa una lista vacia para almacenar las posiciones de los individuos
        self.__Gbest = []      #Se inicializa una lista vacía para almacenar las mejor posicion global 
        self.mejorFitnes = 0   #Se inicializa el mejor fitness a 0 


    #Función que actualiza la mejor solucion del algoritmo.
    #GBest -> Mejor solucion
    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    
    #Convierte a las particulas pasadas (posiciones) en una lista
    def _points(self, agents):
        self.__Positions.append([list(i) for i in agents])


    #Retorna la lista de todas las particulas del algoritmo
    def get_agents(self):
        return self.__Positions

    #Retorna la mejor posicion del algoritmo
    def get_Gbest(self):
        return list(self.__Gbest)
    
    def getMejorFitness(self):
        return self.mejorFitnes
    
    def setMejorFitness(self, fitness):
        self.mejorFitnes=fitness


        