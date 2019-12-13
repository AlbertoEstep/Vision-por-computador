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


''' FUNCIONES P1 '''

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
            kx -
            ky -
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
    imagen = normaliza(imagen)
    return imagen

# Representación en pirámide Gaussiana de 4 niveles de una imagen.
def piramide_gaussiana(imagen, nivel = 4, border = cv2.BORDER_DEFAULT):
    p = [imagen]
    for n in range(nivel):
        # Aplicamos el alisamiento gaussiano con valores de 5 del kernel y 7 de sigma
        p.append(cv2.pyrDown(p[-1], borderType = border))
    return p

''' ***************************************************************************************** '''


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
                if res[i][j] < threshold:
                    res[i][j] = 0
    return res


# Calcula el centro de la matriz
def centro_matriz(m):
    # Se calcula el centro de la matriz
    return m[round(m.shape[0]/2)-1, round(m.shape[0]/2)-1]

# Indica si el centro de la matriz es el máximo de ésta
def es_maximo_centro(m):
    return centro_matriz(m) == np.max(m)


def derivadas(imagen, ksize = 3, sigma = 0, border = cv2.BORDER_DEFAULT):
    imagen = convolution(imagen, sigma = sigma)
    kx = cv2.getDerivKernels(1, 0, ksize)
    ky = cv2.getDerivKernels(0, 1, ksize)
    dx = convolucion(imagen, kx[0], kx[1], border = border)
    dy = convolucion(imagen, ky[0], ky[1], border = border)
    return dx, dy


def orientacion(u):
    if u[0] == 0 and u[1] == 0:
        return 0
    else:
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

'''
def get_orientaciones(imagen, puntos_harris, sigma = 4.5):
    dx, dy = derivadas(imagen, sigma = sigma)
    NO SE QUE HACER
    return puntos_harris
'''


def supresion_no_maximos(m, win_size, nivel_piramide):
    m_ampliada = np.ndarray(shape=(m.shape[0]+2*win_size, m.shape[1]+2*win_size))
    m_ampliada[:, :] = 0
    m_ampliada[win_size:m.shape[0]+win_size, win_size:m.shape[1]+win_size] = m.copy()
    maximos = []
    for fila in range(win_size, m.shape[0]+win_size):
        for columna in range(win_size, m.shape[1]+win_size):
            fini_win = fila - win_size
            ffin_win = fila + win_size
            cini_win = columna - win_size
            cfin_win = columna + win_size
            ventana = m_ampliada[fini_win:ffin_win+1, cini_win:cfin_win+1]
            if es_maximo_centro(ventana):
                maximos.append(cv2.KeyPoint((fila-win_size)*(2**nivel_piramide), (columna-win_size)*(2**nivel_piramide), _size=0, _angle=0))
    return maximos


def get_puntos_harris(imagen, block_size = 9, ksize = 7, threshold = 0, win_size = 5, nivel_piramide = 0):
    matrizH = cv2.cornerEigenValsAndVecs(imagen, block_size, ksize)
    matrizH = cv2.split(matrizH)
    lambda1 = matrizH[0]
    lambda2 = matrizH[1]

    harris = operador_harris(lambda1, lambda2, threshold)
    maximos = supresion_no_maximos(harris, win_size, nivel_piramide-1)
    # Se añade la escala a cada punto
    for i in range(len(maximos)):
        maximos[i].size = nivel_piramide*block_size
    return maximos

def pinta_circulos(imagen, puntos_harris, radius = 1, color = 0):
    for punto in puntos_harris:
        x = int(punto.pt[0])
        y = int(punto.pt[1])
        cv2.circle(imagen, center=(y, x), radius=int(punto.size)*radius, color=color, thickness=2)
    return imagen


def apartado1A():
    # Parametros
    imagen = lee_imagen('imagenes/yosemite/Yosemite1.jpg')
    nivel_piramide = 4

    # Realización
    piramide = piramide_gaussiana(imagen, nivel = nivel_piramide)
    puntos_harris = []
    # Se recorre cada imagen y se le calculan los puntos Harris
    for i, img in enumerate(piramide):
        puntos_harris += get_puntos_harris(img, nivel_piramide = i+1)

     # Se añade la imagen a la lista de imágenes
    #imagen_key = cv2.drawKeypoints(imagen, puntos_harris, np.array([]), color = (0,0,255))
    imagen_key = pinta_circulos(imagen, puntos_harris)
    pinta_imagen(imagen_key)


def apartado2A():
    # Parametros
    imagen1 = lee_imagen('imagenes/Tablero1.jpg')
    imagen2 = lee_imagen('imagenes/Tablero2.jpg')

    # Realización
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(imagen1, None)
    kpts2, desc2 = akaze.detectAndCompute(imagen2, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    matched1 = []
    matched2 = []
    nn_match_ratio = 0.8 # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])
    inliers1 = []
    inliers2 = []
    good_matches = []
    inlier_threshold = 2.5 # Distance threshold to identify inliers with homography check
    for i, m in enumerate(matched1):
        col = np.ones((3,1), dtype=np.float64)
        col[0:2,0] = m.pt
        #col = np.dot(homography, col)
        col /= col[2,0]
        dist = math.sqrt(pow(col[0,0] - matched2[i].pt[0], 2) +\
                    pow(col[1,0] - matched2[i].pt[1], 2))
        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
    res = np.empty((max(imagen1.shape[0], imagen2.shape[0]), imagen1.shape[1]+imagen2.shape[1], 3), dtype=np.uint8)
    imagen_matches = cv2.drawMatches(imagen1, inliers1, imagen2, inliers2, good_matches, None, flags = 0)
    cv2.imwrite("akaze_result.png", res)
    inlier_ratio = len(inliers1) / float(len(matched1))
    print('A-KAZE Matching Results')
    print('*******************************')
    print('# Keypoints 1:                        \t', len(kpts1))
    print('# Keypoints 2:                        \t', len(kpts2))
    print('# Matches:                            \t', len(matched1))
    print('# Inliers:                            \t', len(inliers1))
    print('# Inliers Ratio:                      \t', inlier_ratio)
    cv2.imshow('result', res)
    pinta_imagen(imagen_matches)

def ej1():
    apartado1A()
    #input("Pulsa enter para continuar")

def ej2():
    apartado2A()

def main():
    print("Ejercicio 1")
    #ej1()
    #input("Pulsa enter para continuar")
    ej2()



if __name__ == "__main__":
    main()
