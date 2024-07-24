import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suprime las advertencias de convergencia
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)



""" Funcion que pinta imagen pasada"""
def pintaImagen(cuantizada):
        # Guarda la imagen cuantizada en un archivo
       cv2.imwrite('mandril_cuantizado.tif', cuantizada)

       # Lee la imagen original y la imagen cuantizada
       imagenori = cv2.imread('mandril.tif', cv2.IMREAD_COLOR)
       imagenresu = cv2.imread('mandril_cuantizado.tif', cv2.IMREAD_COLOR)

       #Muestra la imagen original y la imagen cuantizada en ventanas separadas
       cv2.imshow('Imagen Original', imagenori)
       cv2.imshow('Nueva Imagen cuantizada', imagenresu)

       cv2.waitKey(0) #Esperamos a pulsar una tecla
       cv2.destroyAllWindows() #Cerramos

#Funcion que genera una imagen cuantizada usando KMeans
def genera_cuantizada(x,tam_paleta):
       # Leemos la imagen
       img=cv2.imread('mandril.tif', cv2.IMREAD_COLOR)

       # Redimensiona la imagen a una matriz 2D de pixeles (cada fila es un pixel, cada columna es una de las componentes RGB)
       # Modificamos img para ir de punto en punto
       z= img.reshape((-1,3))
       z = np.float32(z)
       

       # Elimina los pixeles duplicados para evitar problemas con KMeans
       z_unique = np.unique(z, axis=0)
    
       # Ajusta el número de clusters al número de pixeles únicos si es menor que tam_paleta
       tam_paleta = min(tam_paleta, len(z_unique))

       # Aplica KMeans a los pixeles únicos para encontrar los clusters (colores representativos)
       k_means = KMeans(n_clusters=tam_paleta,init=x, n_init=1,max_iter=1,algorithm="lloyd").fit(z_unique)
   
       # Guardamos el valor de los centroides y de las etiquetas de cada pixel
       labels = k_means.predict(z)

       # Obtiene los centroides de los clusters (los colores representativos)
       paleta = k_means.cluster_centers_
       # con np.uint8 los convertimos a enteros de 8 bits
       paleta = np.uint8(paleta)

       # Reemplaza cada pixel en la imagen original por el color del cluster al que pertenece
       img_cuantizada = paleta[labels.flatten()]
       img_cuantizada2 = img_cuantizada.reshape((img.shape))
   
       return img_cuantizada2

"""
       Funcion que calcula el fitness de una posicion dada. Aplicado a imagenes, lee una imagen, extrae
       sus datos, y la redimensiona para poder calcular el MSE y retornar el fitness.
       ARGUMENTOS:
              x -> Paleta de colores a usar. (Posicion de particula)
              tam_paleta -> nº de colores a usar.
"""

def genera_cuantizada2(x,tam_paleta):
       # Leemos la imagen
       img=cv2.imread('mandril.tif', cv2.IMREAD_COLOR)
       
       # Redimensiona la imagen a una matriz 2D de pixeles
       z= img.reshape((-1,3))
       z = np.float32(z)
   


       # Eliminar píxeles duplicados
       z_unique = np.unique(z, axis=0)
    
       # Ajusta el número de clusters al número de pixeles únicos si es menor que tam_paleta
       tam_paleta = min(tam_paleta, len(z_unique))
       
       # Aplicar KMeans a los pixeles únicos
       k_means = KMeans(n_clusters=tam_paleta, init=x, n_init=1, max_iter=1, algorithm="lloyd").fit(z_unique)
       
       # Predice los clusters para todos los pixeles en la imagen original
       labels = k_means.predict(z)

       # Obtiene los centroides de los clusters
       paleta = k_means.cluster_centers_
       paleta = np.uint8(paleta)
       
       # Reemplaza cada pixel en la imagen original por el color del cluster al que pertenece
       img_cuantizada = paleta[labels.flatten()]
       img_cuantizada2 = img_cuantizada.reshape((z.shape))
       
       #z = np.array(z)
       #img_cuantizada2 = np.array(img_cuantizada2)
       # Calcula el error cuadrático medio entre la imagen original y la imagen cuantizada
       differences = np.subtract(z, img_cuantizada2)
       squared_differences = np.square(differences)
       return squared_differences.mean()

 
