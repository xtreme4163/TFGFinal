import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error





#numero de colores de la paleta
#tam_paleta = 1024

""" Funcion que pinta imagen pasada"""
def pintaImagen(cuantizada):
       cv2.imwrite('mandril_cuantizado.tif', cuantizada)
       imagenori = cv2.imread('mandril.tif', cv2.IMREAD_COLOR)
       imagenresu = cv2.imread('mandril_cuantizado.tif', cv2.IMREAD_COLOR)
       #Mostramos la imagen
       cv2.imshow('Imagen Original', imagenori)
       cv2.imshow('Nueva Imagen cuantizada', imagenresu)
       cv2.waitKey(0) #Esperamos a pulsar una tecla
       cv2.destroyAllWindows() #Cerramos

def genera_cuantizada(x,tam_paleta):
       # Leemos la imagen
       img=cv2.imread('mandril.tif', cv2.IMREAD_COLOR)
   
       # Modificamos img para ir de punto en punto
       z= img.reshape((-1,3))
       z = np.float32(z)
   
       #Sacamos la informacion del ancho, alto y los canales de la imagen (RGB)
       ancho, alto, canales = img.shape
       tam_imagen = ancho*alto # Calculamos el tamaño de la imagen

   
       paleta = x.reshape(tam_paleta,3)
   
       # etiqueta que relaciona cada pixel con su color correspondiente
       labels = np.zeros(tam_imagen, dtype=np.int8) 
       k_means = KMeans(n_clusters=tam_paleta,init=x, n_init=1,max_iter=1,algorithm="lloyd").fit(z)
   
       # Guardamos el valor de los centroides y de las etiquetas de cada pixel
       labels = k_means.labels_
       paleta = k_means.cluster_centers_
   
       # Generamos la nueva paleta, se usan los centroides encontrados por k-means, estos representan los colores promedio de cada cluster
       # con np.uint8 los convertimos a enteros de 8 bits
       paleta = np.uint8(paleta)
   
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
   
   # Modificamos img para ir de punto en punto
   z= img.reshape((-1,3))
   z = np.float32(z)
   
   #Sacamos la informacion del ancho, alto y los canales de la imagen (RGB)
   ancho, alto, canales = img.shape
   tam_imagen = ancho*alto # Calculamos el tamaño de la imagen

   
   paleta = x.reshape(tam_paleta,3)
   
   # etiqueta que relaciona cada pixel con su color correspondiente
   labels = np.zeros(tam_imagen, dtype=np.int8) 
   k_means = KMeans(n_clusters=tam_paleta,init=x, n_init=1,max_iter=1,algorithm = "lloyd").fit(z)
   
   # Guardamos el valor de los centroides y de las etiquetas de cada pixel
   labels = k_means.labels_
   paleta = k_means.cluster_centers_
   
   paleta = np.uint8(paleta)
   
   
   img_cuantizada = paleta[labels.flatten()]
   img_cuantizada2 = img_cuantizada.reshape((z.shape))
   
   z = np.array(z)
   img_cuantizada2 = np.array(img_cuantizada2)
   differences = np.subtract(z, img_cuantizada2)
   squared_differences = np.square(differences)
   return squared_differences.mean()

 
