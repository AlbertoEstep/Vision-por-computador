# -*- coding: utf-8 -*-
# Author -> Alberto Estepa Fernandez
# Date -> December 2019
# Run: ./p3.py

import cv2
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import copy

# Funcion para leer las imagenes de la P1
def lee_imagen(file_name):
    imagen = cv2.imread(file_name, 0)
    return imagen

# Normalizamos la matriz
def normaliza(imagen):
    max = np.amax(imagen)
    min = np.amin(imagen)
    if max>255 or min<0:
        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                imagen[i][j] = (imagen[i][j]-min)/(max-min) * 255
    return imagen

# Mostramos la imagen
def pinta_imagen(imagen, titulo = "Titulo"):
    imagen = normaliza(imagen)
    imagen = imagen.astype(np.uint8)
    plt.figure(0).canvas.set_window_title(titulo)
    plt.imshow(imagen, cmap = "gray")
    plt.show()

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

'''
def convolution_c(img, sigma = 0, border = cv2.BORDER_DEFAULT):
    img = copy.deepcopy(img)
    kernel_x = cv2.getGaussianKernel(6*sigma+1, sigma)
    kernel_y = kernel_x
    if(len(img.shape) == 2):
        rows, cols = img.shape
    else:
        sys.exit("Imagen no valida")

    for i in range(rows):
        convolucion = cv2.filter2D(img[i, :], -1, kernel_x, borderType=border)
        img[i, :] = [sublist[0] for sublist in convolucion]

    for i in range(cols):
        convolucion = cv2.filter2D(img[:, i], -1, kernel_y, borderType=border)
        img[:, i] = [sublist[0] for sublist in convolucion]

    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    return img
'''

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
def piramide_gaussiana(imagen, nivel = 4, border = cv2.BORDER_DEFAULT):
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
        copia = gaussian_blur(copia, 3, 3, 7, 7, border = border)
        p.append(cv2.pyrDown(copia, borderType = border))
    return p

'''
def piramide_gaussiana(imagen, nivel = 4, border = cv2.BORDER_DEFAULT):

    Una función que genere una representación en pirámide Gaussiana de 4 niveles de una imagen.
        Parametros:
            imagen - Imagen
            nivel – Indica el nivel de la pirámide
            border - Tratamiento del borde de la imagen

    p = [imagen]
    for n in range(nivel):
        # Aplicamos el alisamiento gaussiano con valores de 5 del kernel y 7 de sigma
        p.append(cv2.pyrDown(p[-1], borderType = border))
    return p
'''
'''
def generate_gaussian_pyr_imgs(imagen, niveles = 4, border = cv2.BORDER_DEFAULT , sigma = 0):
    if sigma > 0:
        imagen = convolution_c(imagen, sigma=sigma)
    p = [imagen]
    for i in range(niveles):
        p.append(cv2.pyrDown(p[i], borderType=border))
    return p
'''
# Operador de Harris
def operador_harris(lambda1, lambda2, threshold):
    '''
    Det(matriz)/Traza(matriz)
    '''
    res = np.zeros(lambda1.shape)
    for i in range(lambda1.shape[0]):
        for j in range (lambda1.shape[1]):
            if lambda1[i][j] == 0 and lambda2[i][j] == 0:
                res[i][j] = 0
            else:
                res[i][j] = lambda1[i][j]*lambda2[i][j]/(lambda1[i][j]+lambda2[i][j])
                if res[i][j] <= threshold:
                    res[i][j] = 0
    return res


# Calcula el centro de la matriz
def centro_matriz(m):
    # Se calcula el centro de la matriz
    return m[round(m.shape[0]/2)-1, round(m.shape[0]/2)-1]

# Indica si el centro de la matriz es el máximo de ésta
def es_maximo_centro(m):
    return centro_matriz(m) == np.max(m)
'''
def convolution_d(img, ksize = 3, sigma = 0, border = cv2.BORDER_DEFAULT, dx = 1, dy = 1):
    img = convolution_c(img, sigma = sigma, own = own)
    img_x = copy.deepcopy(img)
    img_y = copy.deepcopy(img)
    kernel_x = cv2.getDerivKernels(dx, 0, ksize)
    kernel_y = cv2.getDerivKernels(0, dy, ksize)
    img_x = convolution_c(img_x, kernel_x[0], kernel_x[1], border = border)
    img_y = convolution_c(img_y, kernel_y[0], kernel_y[1], border = border)
    return img_x, img_y
'''
def orientacion(u):
    u = u / sqrt(u[0]*u[0]+u[1]*u[1])
    if u[1] != 0:
        theta = math.atan(u[0]/u[1])
        if u[0]>0 and u[1]<0:
            theta = math.pi - theta
        elif u[0]<0 and u[1]<0:
            theta = math.pi + theta
        elif u[0]<0 and u[1]>0:
            theta = 2*math.pi - theta
    else:
        if u[0]>0:
            theta = math.pi/2
        else:
            theta = 3/2 * math.pi

    return theta * 180 / math.pi

def get_orientations(img, points, sigma):
    img_x, img_y = convolution_d(img, sigma = sigma)
    new_points = np.array([(int(point[0].pt[0]), int(point[0].pt[1])) for point in points])
    orientations = np.arctan2(img_x, img_y)[new_points[0], new_points[1]]
    for point, orientation in zip(points, orientations):
        point[0].angle = orientation/np.pi*180
    return points

def supresion_no_maximos(m, threshold, win_size):
    min_m = np.min(m)

    m_bordes = np.ndarray(shape=(m.shape[0]+2*win_size, m.shape[1]+2*win_size))
    m_bordes[:, :] = min_m
    m_bordes[win_size:m.shape[0]+win_size, win_size:m.shape[1]+win_size] = m.copy()

    m_auxiliar = np.ndarray(shape=(m.shape[0]+2*win_size, m.shape[1]+2*win_size))
    m_auxiliar[:, :] = 255

    maximos = []
    for row in range(win_size, m.shape[0]+win_size):
        for col in range(win_size, m.shape[1]+win_size):
            if m_auxiliar[row, col] == 255 and m_bordes[row, col]>threshold:
                row_insp_init = row - win_size
                row_insp_end = row + win_size
                col_insp_init = col - win_size
                col_insp_end = col + win_size
                data = m_bordes[row_insp_init:row_insp_end+1, col_insp_init:col_insp_end+1]
                if es_maximo_centro(data):
                    m_auxiliar[row_insp_init:row_insp_end+1, col_insp_init:col_insp_end+1] = 0
                    maximos.append(cv2.KeyPoint(row-win_size, col-win_size, _size=0, _angle=0))
    return maximos

"""
Obtiene una lista potencial de los puntos Harris de una imagen (img).
Los valores de los parámetros utilizados dependen del sigma que se
recibe (sigma_block_size, sigma_ksize).
Recibe (k), (threshol) y (win_size) utilizados en las funciones creadas para la
obtención de los puntos.
Se puede indicar la escala (scale) para devolver la escala a la que
pertenecen los puntos generados junto a estos.
"""
def get_puntos_harris(imagen, block_size = 9, ksize = 7, threshold = 0.1, win_size = 5, nivel_piramide = 0):
    imagen = copy.deepcopy(imagen)
    # Se calculan los autovalores y los autovectores de la imagen
    matrizH = cv2.cornerEigenValsAndVecs(imagen, block_size, ksize)
    # Por cada pixel se tienen 6 valores:
    # - lambda11, lambda2: autovalores
    # - x1, y1: autovalores correspondientes a lambda1
    # - x2, y2: autovalores correspondientes a lambda2
    matrizH = cv2.split(matrizH)
    lambda1 = matrizH[0]
    lambda2 = matrizH[1]
    x1 = matrizH[2]
    y1 = matrizH[3]
    x2 = matrizH[4]
    y2 = matrizH[5]


    # Calculamos el operador de harris
    harris = operador_harris(lambda1, lambda2, threshold)

    # Se suprimen los valores no máximos
    maximos = supresion_no_maximos(harris, threshold, win_size)
    # Se añade la escala a cada punto
    for i in range(len(maximos)):
        maximos[i].size = nivel_piramide*block_size
    # Se devuelven los puntos
    return maximos


'''
def show_circles(img, points, radius = 2, color1 = 0, color2 = 255):
    img = copy.deepcopy(img)
    for point in points:
        x = int(point[0].pt[0])
        y = int(point[0].pt[1])
        size = int(point[0].size)
        cv2.circle(img, center=(y, x), radius=size*radius, color=color1, thickness=2)
    for point in points:
        pt1 = (int(point[0].pt[1]), int(point[0].pt[0]))
        pt2 = (int(point[0].pt[1]+np.sin(point[0].angle)*point[0].size*radius), \
                int(point[0].pt[0]+np.cos(point[0].angle)*point[0].size*radius))
        cv2.line(img, pt1, pt2, color2)
    return img
'''

def apartado1A():
    # Parametros
    imagen = lee_imagen('imagenes/yosemite/Yosemite1.jpg')
    nivel_piramide = 4

    # Realización
    piramide = piramide_gaussiana(imagen, nivel = nivel_piramide)
    puntos_harris = []
    # Se recorre cada imagen y se le calculan los puntos Harris
    for i, imagen in enumerate(piramide):
        puntos_harris = puntos_harris + get_puntos_harris(imagen, nivel_piramide = i+1)

     # Se añade la imagen a la lista de imágenes
    imagen_key = cv2.drawKeypoints(imagen, puntos_harris, np.array([]), color = (0,0,255))
    pinta_imagen(imagen_key)



def ej1():
    apartado1A()
    #input("Pulsa enter para continuar")


def main():
    print("Ejercicio 1")
    ej1()
    #input("Pulsa enter para continuar")


if __name__ == "__main__":
    main()
