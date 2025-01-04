import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import pairwise_distances_argmin
import copy
import warnings
import os
from scipy import signal
from scipy.ndimage import generic_laplace,uniform_filter,correlate,gaussian_filter
from enum import Enum
import phasepack.phasecong as pc  #para fsim


# Suprime las advertencias de convergencia
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)



""" Funcion que pinta imagen pasada"""
def pintaImagen(cuantizada,nombreImagen,pintor,algoritmo,numeroColores):
       # Obtener la ruta del directorio donde se está ejecutando el script
       ruta_script = os.path.dirname(os.path.abspath(__file__))
       # Subir un nivel para obtener el directorio raiz
       ruta_prueba = os.path.dirname(ruta_script)

       #Acceso al directorio de volcado de imagenes cuantizadas
       rutaImagenesCuantizadas = os.path.join(ruta_prueba, 'imagenesCuantizadas')


       # doy nombre a la imagen de salida en formato ALGORITMO_NUMCOLORES_IMAGEN
       nombreSalida = algoritmo + "_" + str(numeroColores) + "_" + os.path.basename(nombreImagen)
       #Contruyo la ruta hacia el directorio de destino
       rutaDestino = os.path.join(rutaImagenesCuantizadas, nombreSalida)

       # Guarda la imagen cuantizada en un archivo
       resultado_guardado = cv2.imwrite(rutaDestino, cuantizada)

        # Verificar si la imagen se guardó correctamente
       if not resultado_guardado:
              print(f"Error: No se pudo guardar la imagen en '{nombreSalida}'")
              return

       # Lee la imagen original y la imagen cuantizada
       imagenori = cv2.imread(nombreImagen, cv2.IMREAD_COLOR)
       imagenresu = cv2.imread(rutaDestino, cv2.IMREAD_COLOR)

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
              # Esta linea no hace falta ya que el propio script elimina la imagen
              #os.remove(nombreSalida)

      

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

#Funcion que genera una imagen cuantizada
def generaCuantizada(gbest,nombreImagen,ajuste):
       #Preparamos imagen
       z,img=preparaImagen(nombreImagen)
       # Obtiene los centroides de los clusters (los colores representativos)
       paleta = copy.deepcopy(gbest)		


       # para cada pixel de la imagen original, se identifica 
       # el número del color de la paleta cuantizada más similar 
       label = pairwise_distances_argmin(paleta, z, axis=0) 
       
       # Se calculan los centroides de los clusters definidos al asociar
       # los pixels de la imagen original con los colores de la paleta cuantizada.
       # Esa información se usa para definir una nueva paleta cuantizada más ajustada
       # a la imagen original (lo que puede mejorar el resultado).
       #
       # promedios -> tantos elementos como colores incluye la paleta cuantizada
       #   Cada elemento será la suma de los colores de los pixels de la imagen
       #   original que se han asociado al mismo elemento de la paleta
       # contadores -> tantos elementos como colores incluye la paleta cuantizada
       #   Cada elemento es el número de sumando almacenados en la posición paralela
       #   de promedios 
       if (ajuste == 1):       
         # Calculamos la media de los puntos en X que corresponden a cada punto en Y.
         #    Si se va a producir ese error, asigno un pixel aletorio de la imagen original 
          # número de pixels de la imagen original ( igual al de elementos de label). CREO QUE NO FUNCIONA BIEN
          npixels = len(label)          
          medias = np.array([z[label == i].mean(axis=0) if z[label == i].size > 0 else z[np.random.randint(0, npixels-1)] for i in range(paleta.shape[0])])    
          
          paleta = copy.deepcopy(medias)

          #poner los dos puntos para que se mofique realmente el contenido del parámetro
          gbest[:] = np.round(medias) # copy.deepcopy(medias)
          	
       
       # antes de convertir a enteros, aplicamos round, para aproximar al entero más próximo
       paleta = np.round(paleta)
       
       # con np.uint8 los convertimos a enteros de 8 bits
       #  con uint8 se truncan los valores reales. Por tanto, quizás sea mejor
       #         primero truncarlos y luego aplicar pairwise
       paleta = np.uint8(paleta)
               
       # por si algún valor se sale del rango válido para RGB, podemos recortarlo
       paleta = np.clip(paleta, 0, 255)
       
       # se generan los pixels de la imagen cuantizada
       res = paleta[label.flatten()]   

       #Se redimensiona el array de pixeles para que coincida con la imagen original
       res2 = res.reshape((img.shape))
     
       return res2

"""
       Funcion que calcula el fitness de una posicion dada. Aplicado a imagenes, lee una imagen, extrae
       sus datos, y la redimensiona para poder calcular el MSE y retornar el fitness.
       ARGUMENTOS:
              x -> Paleta de colores a usar. (Posicion de particula)
              tam_paleta -> nº de colores a usar.
"""
def getMse(x,tam_paleta,nombreImagen, ajuste):
       z,img =preparaImagen(nombreImagen)
       img_cuantizada2 = generaCuantizada(x,nombreImagen, ajuste)

       # Aplanar img_cuantizada2 para que coincida con la forma de z
       img_cuantizada2_flat = img_cuantizada2.reshape((-1, 3))

       # Calcula el error cuadrático medio entre la imagen original y la imagen cuantizada
       return mean_squared_error(z, img_cuantizada2_flat)


""" Igual que la anterior funciçon pero con el Mae (Da menor fitness)
    ERROR ABSOLUTO MEDIO
"""
def getMae(x, tam_paleta,nombreImagen, ajuste):
    # Prepara la imagen, devolviendo tanto la imagen aplanada (z) como la original (img)
    z, img = preparaImagen(nombreImagen)
    
    # Genera la imagen cuantizada
    img_cuantizada2 = generaCuantizada(x,nombreImagen, ajuste)
    
    # Aplanar img_cuantizada2 para que coincida con la forma de z
    img_cuantizada2_flat = img_cuantizada2.reshape((-1, 3))
    
    # Calcular el MAE entre la imagen original (aplanada) y la imagen cuantizada (aplanada)
    mae = np.mean(np.abs(z - img_cuantizada2_flat))
    
    return mae

"""    SSIM índice de similitud estructural
       Devuelve el fitness
"""
def getSsim(x, tam_paleta,nombreImagen, ajuste):
    # Prepara la imagen, devolviendo tanto la imagen aplanada (z) como la original (img)
    z, img = preparaImagen(nombreImagen)
    
    # Genera la imagen cuantizada
    img_cuantizada2 = generaCuantizada(x,nombreImagen, ajuste)
    
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
def getMsSsim(x, tam_paleta,nombreImagen, ajuste):
        
    # Prepara la imagen, devolviendo tanto la imagen aplanada (z) como la original (img)
    z, img = preparaImagen(nombreImagen)

    # Genera la imagen cuantizada usando la paleta de colores (x)
    img_cuantizada2 = generaCuantizada(x,nombreImagen, ajuste)

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

# Calcula el índice UQI para una ventana del par de imágenes (no para
# la imagen completa), cuyo tamaño queda determinado por el parámetro ws
#
# ws: número de pixels de la ventana que se compara
def _uqi_single(GT,P,ws):

        # N es el cuadrado del parámetro ws
	N = ws**2
	
	window = np.ones((ws,ws))

        # para la primera imagen
	GT_sq = GT*GT
	
	# para la segunda imagen
	P_sq = P*P
	
	#para ambas imágenes
	GT_P = GT*P

        # ¿extrae una ventana de tamaño ws de cada una de las dos imagenes de entrada completas?
	GT_sum = uniform_filter(GT, ws)    
	P_sum =  uniform_filter(P, ws)   
	  
        # ¿extrae una ventana de tamaño ws de cada una de las tres matrices producto calculadas antes?
	GT_sq_sum = uniform_filter(GT_sq, ws)  
	P_sq_sum = uniform_filter(P_sq, ws)  
	GT_P_sum = uniform_filter(GT_P, ws)

	GT_P_sum_mul = GT_sum*P_sum
	GT_P_sum_sq_sum_mul = GT_sum*GT_sum + P_sum*P_sum
	
	numerator = 4*(N*GT_P_sum - GT_P_sum_mul)*GT_P_sum_mul
	denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
	denominator = denominator1*GT_P_sum_sq_sum_mul

	q_map = np.ones(denominator.shape)
	index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
	q_map[index] = 2*GT_P_sum_mul[index]/GT_P_sum_sq_sum_mul[index]
	index = (denominator != 0)
	q_map[index] = numerator[index]/denominator[index]

	s = int(np.round(ws/2))
	return np.mean(q_map[s:-s,s:-s])

# Indice UQI.
# Calcula el error medio de los ws errores calculados sobre diversas
#  ventanas de pixels de las imágenes
def getUqi (x,tam_paleta,nombreImagen,ajuste):

	"""calculates universal image quality index (uqi).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).

	:returns:  float -- uqi value.
	"""
	wsPARAM=8 # no se pasa como parámetro
	z, GT =preparaImagen(nombreImagen)
	shapeimg = GT.shape
	
	P = generaCuantizada(x, nombreImagen,ajuste)

	GT = GT.astype(np.float64)
	P = P.astype(np.float64)
	
	indice_UQI =  np.mean([_uqi_single(GT[:,:,i],P[:,:,i],wsPARAM) for i in range(GT.shape[2])])	
	
	return (1.0-indice_UQI)


class Filter(Enum):
	UNIFORM = 0
	GAUSSIAN = 1
	
	
	    
	
def _get_sums(GT,P,win,mode='same'):
	mu1,mu2 = (filter2(GT,win,mode),filter2(P,win,mode))
	return mu1*mu1, mu2*mu2, mu1*mu2

	
def _get_sigmas(GT,P,win,mode='same',**kwargs):
	if 'sums' in kwargs:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = kwargs['sums']
	else:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode)

	return filter2(GT*GT,win,mode)  - GT_sum_sq,\
			filter2(P*P,win,mode)  - P_sum_sq, \
			filter2(GT*P,win,mode) - GT_P_sum_mul
			
				
def filter2(img,fltr,mode='same'):
	return signal.convolve2d(img, np.rot90(fltr,2), mode=mode)
		
def fspecial(fltr,ws,**kwargs):
	if fltr == Filter.UNIFORM:
		return np.ones((ws,ws))/ ws**2
	elif fltr == Filter.GAUSSIAN:
		x, y = np.mgrid[-ws//2 + 1:ws//2 + 1, -ws//2 + 1:ws//2 + 1]
		g = np.exp(-((x**2 + y**2)/(2.0*kwargs['sigma']**2)))
		g[ g < np.finfo(g.dtype).eps*g.max() ] = 0
		assert g.shape == (ws,ws)
		den = g.sum()
		if den !=0:
			g/=den
		return g
	return None	


def _vifp_single(GT,P,sigma_nsq):
	EPS = 1e-10
	num =0.0
	den =0.0
	for scale in range(1,5):
		N=2.0**(4-scale+1)+1
		win = fspecial(Filter.GAUSSIAN,ws=N,sigma=N/5)

		if scale >1:
			GT = filter2(GT,win,'valid')[::2, ::2]
			P = filter2(P,win,'valid')[::2, ::2]

		GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode='valid')
		sigmaGT_sq,sigmaP_sq,sigmaGT_P = _get_sigmas(GT,P,win,mode='valid',sums=(GT_sum_sq,P_sum_sq,GT_P_sum_mul))


		sigmaGT_sq[sigmaGT_sq<0]=0
		sigmaP_sq[sigmaP_sq<0]=0

		g=sigmaGT_P /(sigmaGT_sq+EPS)
		sv_sq=sigmaP_sq-g*sigmaGT_P
		
		g[sigmaGT_sq<EPS]=0
		sv_sq[sigmaGT_sq<EPS]=sigmaP_sq[sigmaGT_sq<EPS]
		sigmaGT_sq[sigmaGT_sq<EPS]=0
		
		g[sigmaP_sq<EPS]=0
		sv_sq[sigmaP_sq<EPS]=0
		
		sv_sq[g<0]=sigmaP_sq[g<0]
		g[g<0]=0
		sv_sq[sv_sq<=EPS]=EPS
		
	
		num += np.sum(np.log10(1.0+(g**2.)*sigmaGT_sq/(sv_sq+sigma_nsq)))
		den += np.sum(np.log10(1.0+sigmaGT_sq/sigma_nsq))

	return num/den

def getVifp(x,tam_paleta,nombreImagen,ajuste):

	"""calculates Pixel Based Visual Information Fidelity (vif-p).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param sigma_nsq: variance of the visual noise (default = 2)

	:returns:  float -- vif-p value.
	"""
	# se convierte el parámetro en una variable local, para homogeneizar 
	# la llamada a las funciones que calculan índices de error
	sigma_nsq=2
	
	z, GT =preparaImagen(nombreImagen)
	shapeimg = GT.shape

	P = generaCuantizada(x, nombreImagen,ajuste)
	
	org_img = GT.astype(np.float64)
	pred_img = P.astype(np.float64)	

	indice_VIF = np.mean([_vifp_single(GT[:,:,i],P[:,:,i],sigma_nsq) for i in range(GT.shape[2])])
	return (1.0 - indice_VIF)	



    
def getFsim(x,tam_paleta,nombreImagen,ajuste):
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)

    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.

    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.

    Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
    are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
    band and then take the average.

    Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
    would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.

    Args:
        org_img -- numpy array containing the original image
        pred_img -- predicted image
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """
    #T1 y T2 dejan de ser parámetros y se convierten en variables dentro de la función
    T1 = 0.85
    T2 = 160
    
    z, GT =preparaImagen(nombreImagen)
    shapeimg = GT.shape

    P = generaCuantizada(x, nombreImagen,ajuste)
	
    org_img = GT.astype(np.float64)
    pred_img = P.astype(np.float64)
	
	    
    # if image is a gray image - add empty 3rd dimension for the .shape[2] to exist
    if org_img.ndim == 2:
        org_img = np.expand_dims(org_img, axis=-1)

    alpha = (
        beta
    ) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(org_img.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(
            org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )
        pc2_2dim = pc(
            pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros(
            (pred_img.shape[0], pred_img.shape[1]), dtype=np.float64
        )
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc**alpha) * (S_g**beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    indice_fsim = np.mean(fsim_list)
    
    return (1.0 - indice_fsim)

def _gradient_magnitude(img: np.ndarray, img_depth: int):
    """
    Calculate gradient magnitude based on Scharr operator.
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx**2 + scharry**2)
    	
def _similarity_measure(x: np.array, y: np.array, constant: float):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x**2 + y**2 + constant

    return numerator / denominator