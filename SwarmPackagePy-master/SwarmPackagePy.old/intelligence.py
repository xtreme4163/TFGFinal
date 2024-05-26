"""
 Clase para definir los métodos usados por todos los algoritmos, 
 como actualizar la g (mejor posicion) ...

"""
class sw(object):

    # Define las posiciones y las mejores posiciones
    def __init__(self):

        self.__Positions = []
        self.__Gbest = []

    #Función que actualiza la mejor solucion del algoritmo.
    #GBest -> Mejor solucion
    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    
    #Convierte a las particulas pasadas (posiciones) en una lista
    def _points(self, agents):
        self.__Positions.append([list(i) for i in agents])

    def get_agents(self):
        """Returns a history of all agents of the algorithm (return type:
        list)"""

        return self.__Positions

    def get_Gbest(self):
        """Return the best position of algorithm (return type: list)"""

        return list(self.__Gbest)
