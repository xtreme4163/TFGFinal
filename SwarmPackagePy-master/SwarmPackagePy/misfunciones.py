import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from skimage.metrics import structural_similarity as ssim
import warnings
import os


# Suprime las advertencias de convergencia
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)



""" Funcion que pinta imagen pasada"""
def pintaImagen(cuantizada,nombreImagen,pintor):
       # Si la imagen es un archivo JPEG distinto nombreSalida
       if nombreImagen.lower().endswith('.jpg') or nombreImagen.lower().endswith('.jpeg'):
              nombreSalida = nombreImagen.split('.')[0] + '-cuantizada.jpg'
       else:
              nombreSalida = nombreImagen.split('.')[0] + '-cuantizada.ppm'

        # Guarda la imagen cuantizada en un archivo
       resultado_guardado = cv2.imwrite(nombreSalida, cuantizada)

        # Verificar si la imagen se guardó correctamente
       if not resultado_guardado:
              print(f"Error: No se pudo guardar la imagen en '{nombreSalida}'")
              return

       # Lee la imagen original y la imagen cuantizada
       imagenori = cv2.imread(nombreImagen, cv2.IMREAD_COLOR)
       imagenresu = cv2.imread(nombreSalida, cv2.IMREAD_COLOR)

        # Si la imagen es un archivo JPEG, redimensionarla
       if nombreImagen.lower().endswith('.jpg') or nombreImagen.lower().endswith('.jpeg'):
              imagenori = redimensionar_imagen(imagenori, 800, 800)
              imagenresu = redimensionar_imagen(imagenresu, 800, 800)

       if(pintor):
              #Muestra la imagen original y la imagen cuantizada en ventanas separadas
              cv2.imshow('Imagen Original', imagenori)
              cv2.imshow('Nueva Imagen cuantizada', imagenresu)

              cv2.waitKey(0) #Esperamos a pulsar una tecla
              cv2.destroyAllWindows() #Cerramos
              # Eliminar la imagen cuantizada, si no baja el fitness, quitar esto, no se porque con esto no consigue bajar... libreria os ?

              os.remove(nombreSalida)

def redimensionar_imagen(imagen, max_width, max_height):
    # Obtener las dimensiones actuales de la imagen
    height, width = imagen.shape[:2]

    # Calcular la relación de aspecto
    aspect_ratio = width / height

    # Determinar las nuevas dimensiones manteniendo la relación de aspecto
    if width > max_width or height > max_height:
        if aspect_ratio > 1:
            # Imagen más ancha que alta
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            # Imagen más alta que ancha
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        # Redimensionar la imagen
        imagen = cv2.resize(imagen, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return imagen
           



#Funcion que prepara una imagen para su posterior uso
def preparaImagen(nombreImagen):
       # Leemos la imagen
       img=cv2.imread(nombreImagen, cv2.IMREAD_COLOR)

       # Si la imagen es un archivo JPEG, redimensionarla
       if nombreImagen.lower().endswith('.jpg') or nombreImagen.lower().endswith('.jpeg'):
              img = redimensionar_imagen(img, 800, 800)

       # Redimensiona la imagen a una matriz 2D de pixeles (cada fila es un pixel, cada columna es una de las componentes RGB)
       # Modificamos img para ir de punto en punto
       z= img.reshape((-1,3))
       z = np.float32(z)

       return z,img

#Funcion que genera una imagen cuantizada usando KMeans
def generaCuantizada(x,tam_paleta,nombreImagen):
       #Preparamos imagen
       z,img=preparaImagen(nombreImagen)
       
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
def getMse(x,tam_paleta,nombreImagen):
       z,img =preparaImagen(nombreImagen)
       img_cuantizada2 = generaCuantizada(x,tam_paleta,nombreImagen)

       # Aplanar img_cuantizada2 para que coincida con la forma de z
       img_cuantizada2_flat = img_cuantizada2.reshape((-1, 3))

       # Calcula el error cuadrático medio entre la imagen original y la imagen cuantizada
       #differences = np.subtract(z, img_cuantizada2_flat)
       #squared_differences = np.square(differences)
       #return squared_differences.mean()
       return mean_squared_error(z, img_cuantizada2_flat)


""" Igual que la anterior funciçon pero con el Mae (Da menor fitness)
    ERROR ABSOLUTO MEDIO
"""
def getMae(x, tam_paleta,nombreImagen):
    # Prepara la imagen, devolviendo tanto la imagen aplanada (z) como la original (img)
    z, img = preparaImagen(nombreImagen)
    
    # Genera la imagen cuantizada
    img_cuantizada2 = generaCuantizada(x, tam_paleta,nombreImagen)
    
    # Aplanar img_cuantizada2 para que coincida con la forma de z
    img_cuantizada2_flat = img_cuantizada2.reshape((-1, 3))
    
    # Calcular el MAE entre la imagen original (aplanada) y la imagen cuantizada (aplanada)
    mae = np.mean(np.abs(z - img_cuantizada2_flat))
    
    return mae

"""    SSIM índice de similitud estructural
       Devuelve el fitness
"""
def getSsim(x, tam_paleta,nombreImagen):
    # Prepara la imagen, devolviendo tanto la imagen aplanada (z) como la original (img)
    z, img = preparaImagen(nombreImagen)
    
    # Genera la imagen cuantizada
    img_cuantizada2 = generaCuantizada(x, tam_paleta,nombreImagen)
    
    # Redimensiona la imagen cuantizada a una matriz 2D de píxeles
    img_cuantizada2_plana = img_cuantizada2.reshape((-1, 3))
    img_cuantizada2_plana = np.float32(img_cuantizada2_plana)
    
    # Calcular el SSIM entre la imagen original y la imagen cuantizada
    ssim_index = ssim(z, img_cuantizada2_plana,multichannel=True,channel_axis=-1,data_range=img.max() - img.min())

    # Utiliza 1 - SSIM como fitness, donde valores más bajos son mejores
    return 1 - ssim_index

"""    MS-SIM- Indice de similitud multi escalar
       Devuelve el fitness.
"""
# Función de fitness basada en MS-SSIM
def getMsSsim(x, tam_paleta,nombreImagen):
        
    # Prepara la imagen, devolviendo tanto la imagen aplanada (z) como la original (img)
    z, img = preparaImagen(nombreImagen)

    # Genera la imagen cuantizada usando la paleta de colores (x)
    img_cuantizada2 = generaCuantizada(x, tam_paleta,nombreImagen)

    # Redimensiona la imagen cuantizada a una matriz 2D de píxeles
    img_cuantizada2_flat = img_cuantizada2.reshape((-1, 3))
    img_cuantizada2_flat = np.float32(img_cuantizada2_flat)
    
    # Ajuste del tamaño de la ventana, se asegura que el tamaño de la ventana win_size sea menor o igual al tamño de la imagen más pequeña
    win_size = min(img.shape[0], img.shape[1], 7)
    
    # Calcular el MS-SSIM entre la imagen original y la imagen cuantizada
    ms_ssim_index = ssim(z, img_cuantizada2_flat, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=win_size, channel_axis=-1, data_range=img.max() - img.min())

    
    # Utilizar 1 - MS-SSIM como fitness, donde valores más bajos son mejores
    fitness = 1 - ms_ssim_index
    
    return fitness
