import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from skimage.metrics import structural_similarity as ssim
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


#Funcion que prepara una imagen para su posterior uso
def preparaImagen():
       # Leemos la imagen
       img=cv2.imread('mandril.tif', cv2.IMREAD_COLOR)

       # Redimensiona la imagen a una matriz 2D de pixeles (cada fila es un pixel, cada columna es una de las componentes RGB)
       # Modificamos img para ir de punto en punto
       z= img.reshape((-1,3))
       z = np.float32(z)

       return z,img

#Funcion que genera una imagen cuantizada usando KMeans
def generaCuantizada(x,tam_paleta):
       #Preparamos imagen
       z,img=preparaImagen()
       
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
def getMse(x,tam_paleta):
       z,img =preparaImagen()
       img_cuantizada2 = generaCuantizada(x,tam_paleta)

       # Aplanar img_cuantizada2 para que coincida con la forma de z
       img_cuantizada2_flat = img_cuantizada2.reshape((-1, 3))

       # Calcula el error cuadrático medio entre la imagen original y la imagen cuantizada
       differences = np.subtract(z, img_cuantizada2_flat)
       squared_differences = np.square(differences)
       return squared_differences.mean()


""" Igual que la anterior funciçon pero con el Mae (Da menor fitness)
    ERROR ABSOLUTO MEDIO
"""
def getMae(x, tam_paleta):
    # Prepara la imagen, devolviendo tanto la imagen aplanada (z) como la original (img)
    z, img = preparaImagen()
    
    # Genera la imagen cuantizada
    img_cuantizada2 = generaCuantizada(x, tam_paleta)
    
    # Aplanar img_cuantizada2 para que coincida con la forma de z
    img_cuantizada2_flat = img_cuantizada2.reshape((-1, 3))
    
    # Calcular el MAE entre la imagen original (aplanada) y la imagen cuantizada (aplanada)
    mae = np.mean(np.abs(z - img_cuantizada2_flat))
    
    return mae

"SSIM índice de similitud estructural"
def getSsim(x, tam_paleta):
    # Prepara la imagen, devolviendo tanto la imagen aplanada (z) como la original (img)
    z, img = preparaImagen()
    
    # Genera la imagen cuantizada
    img_cuantizada2 = generaCuantizada(x, tam_paleta)
    
    # Convierte las imágenes a escala de grises para calcular el SSIM
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cuantizada_gray = cv2.cvtColor(img_cuantizada2, cv2.COLOR_BGR2GRAY)
    
    # Calcular el SSIM entre la imagen original y la imagen cuantizada
    ssim_index = ssim(img_gray, img_cuantizada_gray)

    # Utiliza 1 - SSIM como fitness, donde valores más bajos son mejores
    return 1 - ssim_index

" MS-SIM- Indice de similitud multi escalar"
# Función de fitness basada en MS-SSIM
def getMsSsim(x, tam_paleta):
    # Lee la imagen original
    img = cv2.imread('mandril.tif', cv2.IMREAD_COLOR)
    
    # Genera la imagen cuantizada (asegúrate de que esta función esté implementada)
    img_cuantizada2 = generaCuantizada(x, tam_paleta)
    
    # Asegúrate de que la ventana sea menor o igual al tamaño de la imagen más pequeña
    win_size = min(img.shape[0], img.shape[1], 7)  # Ajuste el tamaño de la ventana según sea necesario
    
    # Calcular el MS-SSIM entre la imagen original y la imagen cuantizada
    ms_ssim_index = ssim(img, img_cuantizada2, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=win_size, channel_axis=2)
    
    # Utilizar 1 - MS-SSIM como fitness, donde valores más bajos son mejores
    fitness = 1 - ms_ssim_index
    
    return fitness
