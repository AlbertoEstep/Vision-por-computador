# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

####################################################### FUNCIONES P0 ##########################

# Funcion para leer las imagenes
def lee_imagen(file_name, flag_color):
    if flag_color == 0 or flag_color == 1:
        return cv2.imread(file_name, flag_color)
    else:
        sys.exit('Imagen no valida')

def normalizar_imagen(imagen):
    if len(imagen.shape) == 2:
        minimo = np.amin(imagen)
        maximo = np.amax(imagen)
        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                imagen[i][j] = 255*(imagen[i][j]-minimo)/(maximo-minimo)
    elif len(imagen.shape) == 3:
        minimo = np.amin(imagen, (0,1))
        maximo = np.amax(imagen, (0,1))
        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                for k in range(imagen.shape[2]):
                    imagen[i][j][k] = 255*(imagen[i][j][k]-minimo[k])/(maximo[k]-minimo[k])
    else:
        sys.exit('Imagen no valida')

def pinta_imagen(imagen, titulo):
    normalizar_imagen(imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    plt.figure(0).canvas.set_window_title(titulo)
    plt.imshow(imagen)
    plt.show()

# Funcion para igualar el tamaño de las imagenes
def igualTam(imagenes):
    # Calculamos el minimo de las filas y de las columnas
    minimo_filas = sys.maxsize
    minimo_columnas = sys.maxsize
    for i in range(len(imagenes)):
        imagen = imagenes[i]
        if(len(imagen) < minimo_filas):
            minimo_filas = len(imagen)
        if(len(imagen[0]) < minimo_columnas):
            minimo_columnas = len(imagen[0])

    # Igualamos el tamaño de las imagenes al ancho y al alto mas pequeño
    for i in range(len(imagenes)):
        imagen = imagenes[i]
        imagenes[i] = cv2.resize(imagen, (minimo_filas, minimo_columnas))
    return imagenes

# Funcion para pintar varias imagenes a la vez con pyplot
def pintar_multiples_imagenes_pyplot(secuencia_imagenes, tituloImagen, filas, columnas, nombre):
    # Primero igualamos el tamaño de todas las imagenes
    # secuencia_imagenes = igualTam(secuencia_imagenes)
    fig = plt.figure(0)
    fig.canvas.set_window_title(nombre)

    for i in range(filas*columnas):
        if (i < len(secuencia_imagenes)):
            plt.subplot(filas, columnas, i+1)
            if (len(np.shape(secuencia_imagenes[i])) == 3):
                img = cv2.cvtColor(secuencia_imagenes[i], cv2.COLOR_BGR2RGB)
                plt.imshow(img)
            else:
                plt.imshow(secuencia_imagenes[i], cmap = 'gray')
            plt.title(tituloImagen[i])
            plt.xticks([])
            plt.yticks([])
    plt.show()
#######################################################################################################

####################################################### FUNCIONES Auxiliares ##########################
'''
# No funciona
def pintar_multiples_imagenes_en_una(secuencia_imagenes, titulo):
    height, weight, z = secuencia_imagenes[0].shape
    # Se crea la nueva Imagen inicializada a ceros
    imagen = np.zeros((height*len(secuencia_imagenes), weight*len(secuencia_imagenes)),  np.uint8)
    # Se concatenan las Imagenes
    for i in range(len(secuencia_imagenes)):
        imagen[:height, :weight] = secuencia_imagenes[i]
    pinta_imagen(imagen, titulo)

'''
#######################################################################################################



'''
#NO NOS DEJAN USARLA
# El cálculo de la convolución de una imagen con una máscara 2D. Usar una Gaussiana 2D (GaussianBlur)
def gaussian_blur(img, size = (0, 0), sigma = 0, border = cv2.BORDER_DEFAULT):
	return cv2.GaussianBlur(img, size, sigma, border)
'''


'''
El cálculo de la convolución de una imagen con una máscara 2D.
    Parametros:
        imagen - imagen
        ksize – Tamaño de la abertura, debe ser impar y positivo.
        sigma – Desviación estándar gaussiana. Si no es positivo, se calcula a partir de ksize como sigma = 0.3 * ((ksize-1) * 0.5 - 1) + 0.8.
        border - tratamiento del borde de la imagen

Podemos observar que a mayor sigma, mayor difuminada queda la imagen. Análogo con el ksize. Podemos tambien poner un sigma grande en algun eje y
el suavizado que se obtiene se aprecia mayor en el eje que tenga el sigma más grande.
'''
def gaussian_blur(imagen, ksize, sigma = 0, border = cv2.BORDER_DEFAULT):
    # La función getGaussianKernel calcula y devuelve la matriz de coeficientes de filtro gaussianos
    gaussian = cv2.getGaussianKernel(ksize, sigma)
    # La función aplica un filtro lineal arbitrario a una imagen.
    imagen = cv2.filter2D(imagen, -1, gaussian, anchor=(-1,-1), borderType = border)
    return imagen


'''
El cálculo de la convolución de una imagen con máscaras 1D dadas por getDerivKernels
    Parametros:
        imagen - imagen
        ksize – Tamaño de la abertura, debe ser impar y positivo.
        sigma – Desviación estándar gaussiana. Si no es positivo, se calcula a partir de ksize como sigma = 0.3 * ((ksize-1) * 0.5 - 1) + 0.8.
        border - tratamiento del borde de la imagen

Podemos observar que a mayor sigma, mayor difuminada queda la imagen. Análogo con el ksize. Podemos tambien poner un sigma grande en algun eje y
el suavizado que se obtiene se aprecia mayor en el eje que tenga el sigma más grande.
'''
def convolucion1D(imagen, dX = 0, dY = 0, ksize = 7, normal = True, border = cv2.BORDER_REPLICATE):
    if ((ksize == 1) or (ksize == 3) or (ksize == 5) or (ksize == 7)):
        # La función calcula y devuelve los coeficientes de filtro para derivadas de imágenes espaciales.
        derivada = cv2.getDerivKernels(dx = dX, dy = dY, ksize = ksize, normalize = normal, ktype = cv2.CV_64F)
        # La función aplica un filtro lineal separable a la imagen.
        return cv2.sepFilter2D(imagen, -1, derivada[0], derivada[1], border)
    else:
        sys.exit('El ksize debe ser 1, 3, 5 o 7')

def maskDerivKernels(img,dx = 1,dy = 1,ksize = 3,border = cv2.BORDER_REPLICATE):
    """
    """
    dxdy = cv2.getDerivKernels(dx,dy,ksize)
    return cv2.sepFilter2D(img,-1,dxdy[0],dxdy[1],border)

# Usar la función Laplacian para el cálculo de la convolución 2D con una máscara normalizada de Laplaciana-de-Gaussiana de tamaño variable.
def laplacian_gaussian(img, sigma = 0, border = cv2.BORDER_DEFAULT, size = (0, 0), k_size = 3, depth = 0, scaler = 1, delt = 50):
    blur = gaussian_blur(img, size, sigma, border)
    return cv2.Laplacian(blur, depth, ksize = k_size, scale = scaler, delta = delt, borderType = border)



# Una función que genere una representación en pirámide Gaussiana de 4 niveles de una imagen.
# Encadena Multiples Imagenes en una Ventana OpenCV para Piramides
def construct_pyramid(imgA , imgB):
    # Se obtienen las filas y columnas
    heightA, weightA = imgA.shape
    heightB, weightB = imgB.shape
    # Se crea la nueva Imagen inicializada a ceros
    imagen = np.zeros((max(heightA, heightB), weightA + weightB),  np.uint8)
    # Se concatenan las Imagenes
    imagen[:heightA, :weightA] = imgA
    imagen[:heightB, weightA:weightA + weightB] = imgB
    return imagen

def gaussian_pyramid(img, level = 4, border = cv2.BORDER_DEFAULT):
    images = imgPyr = img
    for i in range(0, level-1):
        imgPyr = cv2.pyrDown(imgPyr, borderType = border)
        images = construct_pyramid(images, imgPyr)
    return images


# Una función que genere una representación en pirámide Laplaciana de 4 niveles de una imagen.
# Piramide Laplaciana
def laplacian_pyramid(img, level = 4, border = cv2.BORDER_DEFAULT):
    # Piramide Gaussiana
    images = [cv2.pyrDown(img, borderType = border)]
    for i in range(1, level):
        images.append(cv2.pyrDown(images[i-1], borderType = border))
    image = images[-1]
    result = image
    # Piramide Laplaciana
    for i in reversed(images[0:-1]):
        a = cv2.pyrUp(image, dstsize = (i.shape[1], i.shape[0]))
        b = cv2.subtract(a, i)
        result = construct_pyramid(result, b)
        image = i
    return result

def ejercicio1():


    # Ejercicio 1 A 1
    imagen = lee_imagen('imagenes/cat.bmp', 1)
    pintar_multiples_imagenes_pyplot([imagen, gaussian_blur(imagen, 7, 1)], ['cat', 'blur_ksize7_sigma1'], 1, 2, 'GaussianBlur')
    pintar_multiples_imagenes_pyplot([imagen, gaussian_blur(imagen, 7, 7), gaussian_blur(imagen, 7, 11), gaussian_blur(imagen, 7, 21) ], ['cat', 'blur_ksize7_sigma7', 'blur_ksize7_sigma11', 'blur_ksize7_sigma21'], 2, 2, 'GaussianBlur')
    pintar_multiples_imagenes_pyplot([imagen, gaussian_blur(imagen, 5, 7), gaussian_blur(imagen, 9, 7), gaussian_blur(imagen, 15, 7) ], ['cat', 'blur_ksize5_sigma7', 'blur_ksize9_sigma7', 'blur_ksize15_sigma7'], 2, 2, 'GaussianBlur')
    pintar_multiples_imagenes_pyplot([imagen, gaussian_blur(imagen, 7, 7, cv2.BORDER_CONSTANT), gaussian_blur(imagen, 7, 7, cv2.BORDER_REFLECT), gaussian_blur(imagen, 7, 7) ], ['cat', 'blur_border_constant', 'blur_border_reflect', 'blur_default'], 2, 2, 'GaussianBlur')


	# Ejercicio 1 A 2
    pintar_multiples_imagenes_pyplot([imagen, convolucion1D(imagen,1, 1, 1), convolucion1D(imagen, 2, 2, 1)], ['original','convolución1D', 'convolución1D'], 1, 3, 'convolución')

    '''
    # Ejercicio 1 B
	imagen = lee_imagen('imagenes/cat.bmp', 1)
	pintar_multiples_imagenes_pyplot([imagen, laplacian_gaussian(imagen, 1, cv2.BORDER_REPLICATE), laplacian_gaussian(imagen, 1, cv2.BORDER_REFLECT)], ['Original', '1 - Replicate', '1 - Reflect'], 1, 3, 'Laplacian')
	pintar_multiples_imagenes_pyplot([imagen, laplacian_gaussian(imagen, 3, cv2.BORDER_REPLICATE), laplacian_gaussian(imagen, 3, cv2.BORDER_REFLECT)], ['Original', '3 - Replicate', '3 - Reflect'], 1, 3, 'Laplacian')

    '''
def main():

    ejercicio1()


    '''
    # Ejercicio 2 A
    imagen = lee_imagen('imagenes/cat.bmp', 0)
    pinta_imagen(gaussian_pyramid(imagen), "Piramide gaussiana")

    # Ejercicio 2 B
    imagen = lee_imagen('imagenes/cat.bmp', 0)
    pinta_imagen(laplacian_pyramid(imagen), "Piramide Laplaciana")
    '''




if __name__ == "__main__":
    main()
