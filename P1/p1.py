# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

####################################################### FUNCIONES P0 ##########################

# Funcion para leer las imagenes
def lee_imagen(file_name, flag_color):
    if flag_color == 0 or flag_color == 1:
        imagen = cv2.imread(file_name, flag_color)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        imagen = imagen.astype(np.float64)
        return imagen
    else:
        sys.exit('Imagen no valida')

def normalizar_imagen(imagen):
    if len(imagen.shape) == 2:
        minimo = np.amin(imagen)
        maximo = np.amax(imagen)
        if minimo < 0 or maximo > 255:
            for i in range(imagen.shape[0]):
                for j in range(imagen.shape[1]):
                    imagen[i][j] = 255 * (imagen[i][j]- minimo) / (maximo - minimo)
    elif len(imagen.shape) == 3:
        minimo = [np.amin(imagen[:,:,0]), np.amin(imagen[:,:,1]), np.amin(imagen[:,:,2])]
        maximo = [np.amax(imagen[:,:,0]), np.amax(imagen[:,:,1]), np.amax(imagen[:,:,2])]
        if minimo[0] < 0 or minimo[1] < 0 or minimo[2] < 0 or maximo[0] > 255 or maximo[1] > 255 or maximo[2] > 255:
            for i in range(imagen.shape[0]):
                for j in range(imagen.shape[1]):
                    for k in range(imagen.shape[2]):
                        imagen[i][j][k] = 255 * (imagen[i][j][k] - minimo[k]) / (maximo[k] - minimo[k])
    else:
        sys.exit('Imagen no valida')

def pinta_imagen(imagen, titulo):
    normalizar_imagen(imagen)
    imagen = imagen.astype(np.uint8)
    plt.figure(0).canvas.set_window_title(titulo)
    plt.imshow(imagen)
    plt.show()
    imagen = imagen.astype(np.float64)

# Funcion para pintar varias imagenes a la vez con pyplot
def pintar_multiples_imagenes_pyplot(secuencia_imagenes, tituloImagen, filas, columnas, nombre):
    fig = plt.figure(0)
    fig.canvas.set_window_title(nombre)
    for i in range(len(secuencia_imagenes)):
        normalizar_imagen(secuencia_imagenes[i])
        secuencia_imagenes[i] = secuencia_imagenes[i].astype(np.uint8)
    for i in range(columnas*filas):
        if i < len(secuencia_imagenes):
            plt.subplot(filas, columnas, i+1)
            plt.imshow(secuencia_imagenes[i])
            plt.title(tituloImagen[i])
            plt.xticks([])
            plt.yticks([])
    plt.show()
    for i in range(len(secuencia_imagenes)):
        secuencia_imagenes[i] = secuencia_imagenes[i].astype(np.float64)


#######################################################################################################

####################################################### FUNCIONES AUXILIARES ##########################

# Eliminamos las filas y columnas impares
def subsampling(imagen):
    return imagen[::2, ::2]

# Imprimimos varias imagenes en una sola
def imprime_varias_imagenes_en_una(secuencia_imagenes, titulo):
    altura = max(imagen.shape[0] for imagen in secuencia_imagenes)
    for i,imagen in enumerate(secuencia_imagenes):
        if imagen.shape[0] < altura:
            secuencia_imagenes[i] = cv2.copyMakeBorder(secuencia_imagenes[i], 0, altura - secuencia_imagenes[i].shape[0], 0, 0, borderType = cv2.BORDER_CONSTANT)
    imagen = cv2.hconcat(secuencia_imagenes)
    pinta_imagen(imagen, titulo)

# Igualamos dos imagenes (nos servira para el bonus 3)
def igualar(imagen_fbaja, imagen_falta):
    altura = min(imagen_fbaja.shape[0], imagen_falta.shape[0])
    ancho = min(imagen_fbaja.shape[1], imagen_falta.shape[1])
    imagen_fbaja = cv2.resize(imagen_fbaja, (ancho, altura), imagen_fbaja)
    imagen_falta = cv2.resize(imagen_falta, (ancho, altura), imagen_falta)
    return imagen_fbaja, imagen_falta

#######################################################################################################

####################################################### FUNCIONES P1 ##########################

'''
Calculo de la convolución una máscara arbitraria.
    Parametros:
        imagen - Imagen
        kx: kernel del eje x.
        ky: kernel del eje y.
        border - Tratamiento del borde de la imagen
'''
def convolucion(imagen, kx, ky, border = cv2.BORDER_DEFAULT):
    # Transponemos el vertor del eje x (para que al multiplicarlo con el vertor del eje y nos resulte una matriz)
    kx = np.transpose(kx)
    # Revertimos los vectores oara la convolución
    kx = cv2.flip(kx, -1)
    ky = cv2.flip(ky, -1)
    # Convoluciona una imagen con el núcleo
    imagen = cv2.filter2D(imagen, -1, kx, borderType = border)
    imagen = cv2.filter2D(imagen, -1, ky, borderType = border)
    return imagen


'''
El cálculo de la convolución de una imagen con una máscara 2D.
    Parametros:
        imagen - Imagen
        kx - Tamaño de la abertura en el eje x, debe ser impar y positivo.
        ky - Tamaño de la abertura en el eje y, debe ser impar y positivo.
        sigmax – Desviación estándar gaussiana en el eje x.
        sigmay – Desviación estándar gaussiana en el eje y.
        border - Tratamiento del borde de la imagen

Podemos observar que a mayor sigma, mayor difuminada queda la imagen. Análogo con el ksize. Podemos tambien poner un sigma grande en algun eje y
el suavizado que se obtiene se aprecia mayor en el eje que tenga el sigma más grande.
'''
def gaussian_blur(imagen, kx = 0, ky = 0, sigmax = 0, sigmay = 0, border = cv2.BORDER_DEFAULT):
    # Calculamos el tamaño de la mascara optimo si el usuario introdujo el valor 0
    if kx == 0:
        # 6 veces el valor del sigma es el tamaño optimo para el tamaño de la abertura, mas 1 para conseguir la imparidad
        kx = int(6*sigmax) + 1
    if ky == 0:
        ky = int(6*sigmay) + 1
    # La función getGaussianKernel calcula y devuelve la matriz de coeficientes de filtro gaussianos
    kernelX = cv2.getGaussianKernel(kx, sigmax)
    kernelY = cv2.getGaussianKernel(ky, sigmay)
    # Hacemos la convolucion
    imagen = convolucion(imagen, kernelX, kernelY, border)
    return imagen

'''
El cálculo de la convolución de una imagen con máscaras 1D dadas por getDerivKernels
    Parametros:
        imagen - Imagen
        dX – Orden de la derivada con respecto al eje X.
        dY – Orden de la derivada con respecto al eje Y.
        ksize – Tamaño de la abertura, debe ser impar y positivo (1, 3, 5 o 7, segun https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#void%20getDerivKernels(OutputArray%20kx,%20OutputArray%20ky,%20int%20dx,%20int%20dy,%20int%20ksize,%20bool%20normalize,%20int%20ktype)).
        border - Tratamiento del borde de la imagen

Observamos cambios de tono de la imagen. Al aumentar el orden de la derivada, se aprecia los cambios de tonos mas fuertes.
Si aumentamos la derivada en algun eje con respecto el otro, aparecen lineas en dirección al eje aumentado.
'''
def convolucion1D(imagen, dX = 0, dY = 0, ksize = 7, border = cv2.BORDER_REPLICATE):
    if ((ksize == 1) or (ksize == 3) or (ksize == 5) or (ksize == 7)):
        # La función calcula y devuelve los coeficientes de filtro para derivadas de imágenes espaciales.
        derivada = cv2.getDerivKernels(dx = dX, dy = dY, ksize = ksize, ktype = cv2.CV_64F)
        imagen = convolucion(imagen, derivada[0], derivada[1], border)
        return imagen
    else:
        sys.exit('El ksize debe ser 1, 3, 5 o 7')

'''
Usar la función Laplacian para el cálculo de la convolución 2D con una máscara normalizada de Laplaciana-de-Gaussiana de tamaño variable.
    Parametros:
        imagen - Imagen
        border - Tratamiento del borde de la imagen
        ksize – Tamaño de la abertura, debe ser impar y positivo.
EXPLICAR
'''
def laplaciana_de_gaussiana(imagen, border = cv2.BORDER_DEFAULT, ksize = 5):
    # La función calcula y devuelve los coeficientes de filtro para derivadas (de orden 2 en el eje x) de imágenes espaciales.
    derivada2x0y = cv2.getDerivKernels(2, 0, ksize)
    # La función calcula y devuelve los coeficientes de filtro para derivadas (de orden 2 en el eje y) de imágenes espaciales.
    derivada0x2y = cv2.getDerivKernels(0, 2, ksize)
    return convolucion(image, derivada2x0y[0], derivada2x0y[1], border) + convolucion(image, derivada0x2y[0], derivada0x2y[1], border)

'''
Una función que genere una representación en pirámide Gaussiana de 4 niveles de una imagen.
    Parametros:
        imagen - Imagen
        nivel – Indica el nivel de la pirámide
        border - Tratamiento del borde de la imagen

EXPLICAR Y MAS EJEMPLOS
'''

# Una función que genere una representación en pirámide Gaussiana de 4 niveles de una imagen.
def piramide_gaussiana(imagen, nivel = 4, border = cv2.BORDER_CONSTANT):
    p = [imagen]
    copia = np.copy(imagen)
    for n in range(nivel):
        # Aplicamos el alisamiento gaussiano
        copia = gaussian_blur(copia, 5, 5, 7, 7, border = border)
        # Aplicamos el subsampling
        copia = subsampling(copia)
        # Unimos el resultado a la piramide
        p.append(copia)
    return p

'''
Una función que genere una representación en pirámide Laplaciana de 4 niveles de una imagen. Mostrar ejemplos de
funcionamiento usando bordes.
    Parametros:
        imagen - Imagen
        nivel – Indica el nivel de la pirámide
        border - Tratamiento del borde de la imagen

EXPLICAR Y MAS EJEMPLOS

def laplacian_pyramid(imagen, nivel = 4, border = cv2.BORDER_CONSTANT):

'''

'''
Escribir una función que muestre las tres imágenes ( alta, baja e híbrida) en una misma ventana. (Recordar que las
imágenes después de una convolución contienen número flotantes que pueden ser positivos y negativos)
    Parametros:
        imagen_fbaja - Imagen usada para frecuencias bajas
        sigma_fbaja: Sigma para la imagen de frecuencias bajas.
        imagen_falta - Imagen usada para frecuencias altas
        sigma_falta: Sigma para la imagen de frecuencias altas.

EXPLICAR
'''
def hibrida(imagen_fbaja, sigma_fbaja, imagen_falta, sigma_falta):
    # Obtenemos las frecuencias bajas de la imagen
    f_bajas = gaussian_blur(imagen_fbaja, sigmax = sigma_fbaja, sigmay = sigma_fbaja)
    # Obtenemos las frecuencias altas de la imagen
    f_altas = cv2.subtract(imagen_falta, gaussian_blur(imagen_falta, sigma_falta, sigma_falta))
    # Calculamos la suma ponderada de las dos matrices con addWeighted con alpha = 0.5, beta = 0.5 y gamma = 0
    hibrida = cv2.addWeighted(f_bajas, 0.5, f_altas, 0.5, 0)
    return [f_bajas, f_altas, hibrida]


def ejercicio1():
    # Ejercicio 1 A 1
    imagen = lee_imagen('imagenes/cat.bmp', 1)
    pintar_multiples_imagenes_pyplot([imagen, gaussian_blur(imagen, 7, 7, 7, 7), gaussian_blur(imagen, 7, 7, 11, 11), gaussian_blur(imagen, 7, 7, 21, 21) ], ['cat', 'blur_ksize7_sigma7', 'blur_ksize7_sigma11', 'blur_ksize7_sigma21'], 2, 2, 'GaussianBlur')
    pintar_multiples_imagenes_pyplot([imagen, gaussian_blur(imagen, 5, 5, 7, 7), gaussian_blur(imagen, 9, 9, 7, 7), gaussian_blur(imagen, 15, 15, 7, 7) ], ['cat', 'blur_ksize5_sigma7', 'blur_ksize9_sigma7', 'blur_ksize15_sigma7'], 2, 2, 'GaussianBlur')
    pintar_multiples_imagenes_pyplot([imagen, gaussian_blur(imagen, 7, 7, 7, 7, cv2.BORDER_CONSTANT), gaussian_blur(imagen, 7, 7, 7, 7, cv2.BORDER_REFLECT), gaussian_blur(imagen, 7, 7, 7, 7) ], ['cat', 'blur_border_constant', 'blur_border_reflect', 'blur_default'], 2, 2, 'GaussianBlur')

	# Ejercicio 1 A 2
    imagen = lee_imagen('imagenes/cat.bmp', 1)
    pintar_multiples_imagenes_pyplot([convolucion1D(imagen, 1, 1, 1), convolucion1D(imagen, 2, 2, 1)], ['dx1_dy1_ksize1', 'dx2_dy2_ksize1'], 1, 2, 'Convolucion1D: Variación en el orden de las derivadas')
    pintar_multiples_imagenes_pyplot([convolucion1D(imagen, 1, 1, 3), convolucion1D(imagen, 1, 1, 5)], ['dx1_dy1_ksize3', 'dx1_dy1_ksize5'], 1, 2, 'Convolucion1D: Variacion en el tamaño de la abertura')
    pintar_multiples_imagenes_pyplot([convolucion1D(imagen, 0, 2, 3), convolucion1D(imagen, 2, 0, 3)], ['dx0_dy2_ksize3', 'dx2_dy0_ksize3'], 1, 2, 'Convolucion1D: Variación en los ejes de las derivadas')
    pintar_multiples_imagenes_pyplot([convolucion1D(imagen, 1, 1, 3, border = cv2.BORDER_CONSTANT), convolucion1D(imagen, 1, 1, 3, border = cv2.BORDER_REFLECT)], ['BORDER_CONSTANT', 'BORDER_REFLECT'], 1, 2, 'Convolucion1D: Variación en los ejes de las derivadas')

    # Ejercicio 1 B
    imagen = lee_imagen('imagenes/cat.bmp', 1)
    pintar_multiples_imagenes_pyplot([imagen, laplaciana_de_gaussiana(imagen, 1, cv2.BORDER_REPLICATE), laplaciana_de_gaussiana(imagen, 1, cv2.BORDER_REFLECT)], ['Original', '1 - Replicate', '1 - Reflect'], 1, 3, 'Laplacian')
    pintar_multiples_imagenes_pyplot([imagen, laplaciana_de_gaussiana(imagen, 3, cv2.BORDER_REPLICATE), laplaciana_de_gaussiana(imagen, 3, cv2.BORDER_REFLECT)], ['Original', '3 - Replicate', '3 - Reflect'], 1, 3, 'Laplacian')

def ejercicio2():
    # Ejercicio 2 A
    imagen = lee_imagen('imagenes/cat.bmp', 1)
    imprime_varias_imagenes_en_una(piramide_gaussiana(imagen, 4, cv2.BORDER_DEFAULT), "Piramide gaussiana")
    imprime_varias_imagenes_en_una(piramide_gaussiana(imagen, 4, cv2.BORDER_CONSTANT), "Piramide gaussiana borde constante")
    imprime_varias_imagenes_en_una(piramide_gaussiana(imagen, 4, cv2.BORDER_REFLECT), "Piramide gaussiana borde reflect")

    # Ejercicio 2 B
    imagen = lee_imagen('imagenes/cat.bmp', 0)

    # SIN HACER imprime_varias_imagenes_en_una(laplacian_pyramid(imagen), "Piramide Laplaciana")

def ejercicio3():

    bici = lee_imagen("imagenes/bicycle.bmp", 0)
    moto = lee_imagen("imagenes/motorcycle.bmp", 0)
    bici_moto = hibrida(bici, 9, moto, 9)
    imprime_varias_imagenes_en_una(bici_moto, 'Bici - Moto')

    einstein = lee_imagen("imagenes/einstein.bmp", 0)
    marilyn = lee_imagen("imagenes/marilyn.bmp", 0)
    einstein_marilyn = hibrida(einstein, 3, marilyn, 7)
    imprime_varias_imagenes_en_una(einstein_marilyn, 'Einstein - Marilyn')

    avion = lee_imagen("imagenes/plane.bmp", 0)
    pajaro = lee_imagen("imagenes/bird.bmp", 0)
    avion_pajaro = hibrida(avion, 5, pajaro, 15)
    imprime_varias_imagenes_en_una(avion_pajaro, 'Avion - Pajaro')

    '''
    Construir pirámides gaussianas de al menos 4 níveles con
    las imágenes resultado. Explicar el efecto que se observa.
    '''

    piramide_bici_moto = piramide_gaussiana(bici_moto[2], 4)
    imprime_varias_imagenes_en_una(piramide_bici_moto, 'Piramide gaussiana de la bici y la moto')

    piramide_einstein_marilyn = piramide_gaussiana(einstein_marilyn[2], 4)
    imprime_varias_imagenes_en_una(piramide_einstein_marilyn, 'Piramide gaussiana de Einstein y la Marilyn')

    piramide_avion_pajaro = piramide_gaussiana(avion_pajaro[2], 4)
    imprime_varias_imagenes_en_una(piramide_avion_pajaro, 'Piramide gaussiana de Einstein y la Marilyn')

def bonus2():
    bici = lee_imagen("imagenes/bicycle.bmp", 1)
    moto = lee_imagen("imagenes/motorcycle.bmp", 1)
    bici_moto = hibrida(bici, 9, moto, 9)
    imprime_varias_imagenes_en_una(bici_moto, 'Bici - Moto')

    einstein = lee_imagen("imagenes/einstein.bmp", 1)
    marilyn = lee_imagen("imagenes/marilyn.bmp", 1)
    einstein_marilyn = hibrida(einstein, 3, marilyn, 5)
    imprime_varias_imagenes_en_una(einstein_marilyn, 'Einstein - Marilyn')

    avion = lee_imagen("imagenes/plane.bmp", 1)
    pajaro = lee_imagen("imagenes/bird.bmp", 1)
    avion_pajaro = hibrida(avion, 5, pajaro, 11)
    imprime_varias_imagenes_en_una(avion_pajaro, 'Avion - Pajaro')

    gato = lee_imagen("imagenes/cat.bmp", 1)
    perro = lee_imagen("imagenes/dog.bmp", 1)
    gato_perro = hibrida(gato, 5, perro, 15)
    imprime_varias_imagenes_en_una(gato_perro, 'Gato - Perro')

    pez = lee_imagen("imagenes/fish.bmp", 1)
    submarino = lee_imagen("imagenes/submarine.bmp", 1)
    pez_submarino = hibrida(pez, 5, submarino, 15)
    imprime_varias_imagenes_en_una(pez_submarino, 'Pez - Submarino')

def bonus3():
    billar = lee_imagen("imagenes/pelota_billar.jpg", 1)
    futbol = lee_imagen("imagenes/pelota_futbol.jpg", 1)
    billar, futbol = igualar(billar, futbol)
    billar_futbol = hibrida(billar, 5, futbol, 15)
    imprime_varias_imagenes_en_una(billar_futbol, 'Billar - Futbol')

def main():
    ejercicio1()
    ejercicio2()
    ejercicio3()
    bonus2()
    bonus3()

if __name__ == "__main__":
    main()
