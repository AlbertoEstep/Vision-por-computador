# -*- coding: utf-8 -*-
# Author -> Alberto Estepa Fernandez
# Date -> December 2019
# Run: ./p3.py

import cv2
import numpy as np
import sys

# Funcion para leer las imagenes de la P1
def lee_imagen(file_name):
    imagen = cv2.imread(file_name)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    imagen = imagen.astype(np.float64)
    return imagen

# Indica si la imagen está en escala de grises
def esBlancoNegro(imagen):
    return len(imagen.shape) == 2

# Calculo de la convolución una máscara arbitraria.
def convolucion(imagen, kx, ky, border = cv2.BORDER_DEFAULT):
    '''
    Calculo de la convolución una máscara arbitraria.
        Parametros:
            imagen - Imagen
            kx: kernel del eje x.
            ky: kernel del eje y.
            border - Tratamiento del borde de la imagen
    '''
    # Transponemos el vertor del eje x (para que al multiplicarlo con el vertor del eje y nos resulte una matriz)
    kx = np.transpose(kx)
    # Revertimos los vectores oara la convolución
    kx = cv2.flip(kx, -1)
    ky = cv2.flip(ky, -1)
    # Convoluciona una imagen con el núcleo
    imagen = cv2.filter2D(imagen, -1, kx, borderType = border)
    imagen = cv2.filter2D(imagen, -1, ky, borderType = border)
    return imagen

# El cálculo de la convolución de una imagen con una máscara 2D.
def gaussian_blur(imagen, kx = 0, ky = 0, sigmax = 0, sigmay = 0, border = cv2.BORDER_DEFAULT):
    '''
    El cálculo de la convolución de una imagen con una máscara 2D.
        Parametros:
            imagen - Imagen
            kx - Tamaño de la abertura en el eje x, debe ser impar y positivo.
            ky - Tamaño de la abertura en el eje y, debe ser impar y positivo.
            sigmax – Desviación estándar gaussiana en el eje x.
            sigmay – Desviación estándar gaussiana en el eje y.
            border - Tratamiento del borde de la imagen
    '''
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

# Representación en pirámide Gaussiana de 4 niveles de una imagen.
def piramide_gaussiana(imagen, nivel = 4, border = cv2.BORDER_CONSTANT):
    '''
    Una función que genere una representación en pirámide Gaussiana de 4 niveles de una imagen.
        Parametros:
            imagen - Imagen
            nivel – Indica el nivel de la pirámide
            border - Tratamiento del borde de la imagen
    '''
    p = [imagen]
    copia = np.copy(imagen)
    for n in range(nivel):
        # Aplicamos el alisamiento gaussiano con valores de 5 del kernel y 7 de sigma
        copia = gaussian_blur(copia, 5, 5, 7, 7, border = border)
        # Aplicamos el subsampling
        p.append(cv2.pyrDown(copia, borderType = border))
    return p

# Operador de Harris
def operador_harris(lambda1, lambda2):
    '''
    Det(matriz)/Traza(matriz)
    '''
    return (lambda1*lambda2)/(lambda1+lambda2)

# Calcula el centro de la matriz
def centro_matriz(m):
    # Se calcula el centro de la matriz
    return m[round(m.shape[0]/2)-1, round(m.shape[0]/2)-1]

# Indica si el centro de la matriz es el máximo de ésta
def es_maximo_centro(m):
    return centro_matriz(m) == np.max(m)


def supresion_no_maximos(m, threshold, env):
    """
    Suprime los valores no-máximos de una matriz (harris). Elimina los puntos
    que, aunque tengan un valor alto de criterio Harris, no son máximos locales
    de su entorno (env) para un tamaño de entorno prefijado.
    Permite la utilización de un umbral (threshold) para eliminar aquellos
    puntos Harris que se consideren elevados.
    Devuelve una lista con los keyPoints y su valor correspondiente.
    """
    # Se consigue el mínimo valor para rellenar los bordes de la matriz con él
    min_m = np.min(m)
    # Se crea una nueva matriz donde se copiarán los puntos harris con un borde
    # de tamaño env relleno con el valor mínimo de los puntos Harris para que
    # no entre en conflicto al calcular el máximo local
    m_bordes = np.ndarray(shape=(m.shape[0]+2*env, m.shape[1]+2*env))
    m_bordes[:, :] = min_m
    m_bordes[env:m.shape[0]+env, env:m.shape[1]+env] = m.copy()
    # Se crea una matriz rellena completamente a 255 con las mismas dimensiones
    # que la imagen actual
    m_auxiliar = np.ndarray(shape=(m.shape[0]+2*env, m.shape[1]+2*env))
    m_auxiliar[:, :] = 255
    # Los puntos Harris recibidos pueden tener multitud de valores. Si se
    # quieren obtener los más representativos se deben eliminar los que no
    # tienen un valor alto. En una zona completamente blanca pueden encontrarse
    # máximos locales, por lo que aparecerían en la imagen. Para evitar esto,
    # se indica un umbral, por el que por debajo de él no se tienen en cuenta
    # esos puntos

    # Lista con las coordenadas de los máximos locales y el valor que tiene
    # cada uno
    maximos = []
    # Se recorre cada posición de la matriz de puntos Harris (se ignora el
    # borde creado manualmente)
    for row in range(env, m.shape[0]+env):
        for col in range(env, m.shape[1]+env):
            # Si el punto actual tiene valor 255 o si su punto Harris está por
            # encima del umbral indicado se comprueba si es un máximo local
            if m_auxiliar[row, col] == 255 and m_bordes[row, col]>threshold:
                # Se obtiene el rango de datos que se deben conseguir para
                # comprobar que es máximo local, teniendo en cuenta el valor
                # de env
                row_insp_init = row - env
                row_insp_end = row + env
                col_insp_init = col - env
                col_insp_end = col + env
                # Se obtiene la matriz con los datos que rodean al punto actual
                data = m_bordes[row_insp_init:row_insp_end+1, \
                                    col_insp_init:col_insp_end+1]
                # Se comprueba si el punto central es máximo local
                if es_maximo_centro(data):
                    # En caso de ser máximo local, todo el rango de datos
                    # seleccionados se cambian a 0 para no volver a comprobarlos
                    m_auxiliar[row_insp_init:row_insp_end+1, \
                                col_insp_init:col_insp_end+1] = 0
                    # Se guarda el punto actual real como máximo local
                    # utilizando la estructura cv2.KeyPoint
                    maximos.append([cv2.KeyPoint(row-env, col-env, _size=0, \
                                    _angle=0), m[row-env, col-env]])
    # Se devuelven las coordenadas de los puntos máximos locales y su valor
    return maximos

"""
Obtiene una lista potencial de los puntos Harris de una imagen (img).
Los valores de los parámetros utilizados dependen del sigma que se
recibe (sigma_block_size, sigma_ksize).
Recibe (k), (threshol) y (env) utilizados en las funciones creadas para la
obtención de los puntos.
Se puede indicar la escala (scale) para devolver la escala a la que
pertenecen los puntos generados junto a estos.
"""
def get_puntos_harris(imagen, block_size = 9, ksize = 7, threshold = -10000, env = 5, scale = -1):
    # Se calculan los autovalores y los autovectores de la imagen
    matrizH = cv2.cornerEigenValsAndVecs(imagen, block_size, ksize)
    # Por cada pixel se tienen 6 valores:
    # - lambda11, lambda2: autovalores
    # - x1, y1: autovalores correspondientes a lambda1
    # - x2, y2: autovalores correspondientes a lambda2
    matrizH = cv2.split(matrizH)
    lambda1 = matrizH[0]
    lambda2 = matrizH[1]

    # Calculamos el operador de harris
    harris = operador_harris(lambda1, lambda2)
    # Se suprimen los valores no máximos
    maximos = supresion_no_maximos(harris, threshold, env)

    # Se añade la escala a cada punto
    for i in range(len(maximos)):
        maximos[i][0].size = scale

    # Se devuelven los puntos
    return maximos


def apartado1A():
    imagen = lee_imagen('imagenes/yosemite/Yosemite1.jpg')
    piramide = piramide_gaussiana(imagen, niveles = 4)
    puntos_harris = []
    # Se recorre cada imagen y se le calculan los puntos Harris
    for i, imagen in enumerate(piramide):
        puntos_harris = puntos_harris + get_harris(imagen, scale = i+1)


def ej1():
    apartado1A()
    input("Pulsa enter para continuar")


def main():
    print("Ejercicio 1")
    ej1()
    input("Pulsa enter para continuar")


if __name__ == "__main__":
    main()
