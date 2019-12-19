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
import random


''' FUNCIONES P1 '''

# Funcion para leer las imagenes de la P1
def lee_imagen(file_name):
    imagen = cv2.imread(file_name, 0)
    imagen = imagen.astype(np.float32)
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
    imagen = imagen.astype(np.uint8)
    plt.figure(0).canvas.set_window_title(titulo)
    plt.imshow(imagen, cmap = "gray")
    plt.show()

def show_images(imgs, cols = 1, title = ""):
    plt.rcParams['image.cmap'] = 'gray'
    imgs_length = len(imgs)
    rows = 1
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
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
    return imagen

def gaussian_blur(imagen, kx = 0, ky = 0, sigmax = 0, sigmay = 0, border = cv2.BORDER_DEFAULT):
    if kx == 0:
        kx = int(6*sigmax) + 1
    if ky == 0:
        ky = int(6*sigmay) + 1
    kernelX = cv2.getGaussianKernel(kx, sigmax)
    kernelY = cv2.getGaussianKernel(ky, sigmay)
    imagen = convolucion(imagen, kernelX, kernelY, border)
    return imagen


# Representación en pirámide Gaussiana de 4 niveles de una imagen.
def piramide_gaussiana(imagen, nivel = 4, border = cv2.BORDER_DEFAULT):
    p = [imagen]
    for n in range(nivel-1):
        # Aplicamos el alisamiento gaussiano con valores de 5 del kernel y 7 de sigma
        p.append(cv2.pyrDown(p[-1], borderType = border))
    return p

def derivadas(imagen, ksize = 3, sigma = 0, border = cv2.BORDER_DEFAULT):
    imagen = gaussian_blur(imagen, kx = ksize, ky = ksize, sigmax = sigma, sigmay = sigma)
    kx = cv2.getDerivKernels(1, 0, ksize)
    ky = cv2.getDerivKernels(0, 1, ksize)
    dx = convolucion(imagen, kx[0], kx[1], border = border)
    dy = convolucion(imagen, ky[0], ky[1], border = border)
    return dx, dy

''' ***************************************************************************************** '''


# Operador de Harris
def operador_harris(lambda1, lambda2):
    '''
    Det(matriz)/Traza(matriz)
    '''
    res = np.zeros(lambda1.shape)
    for i in range(lambda1.shape[0]):
        for j in range (lambda1.shape[1]):
            if lambda1[i][j]+lambda2[i][j] == 0:
                res[i][j] = 0
            else:
                res[i][j] = lambda1[i][j]*lambda2[i][j]/(lambda1[i][j]+lambda2[i][j])
    return res


# Calcula el centro de la matriz
def centro_matriz(m):
    # Se calcula el centro de la matriz
    return m[int(m.shape[0]/2), int(m.shape[1]/2)]

# Indica si el centro de la matriz es el máximo de ésta
def es_maximo_centro(m):
    return centro_matriz(m) == np.max(m)

def orientacion(u1, u2):
    # Normalizamos el vector
    l2_norm = math.sqrt(u1*u1+u2*u2)
    if (l2_norm != 0 ):
        u1 = u1 / l2_norm
        u2 = u2 / l2_norm

        theta = math.atan2(u2, u1) * 180 / math.pi
        if theta < 0:
            theta += 360
    else:
        theta = 0
    # Devolvemos en grados
    return theta

def supresion_no_maximos(imagen, dx, dy, m, win_size, nivel_piramide, block_size, threshold):
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
            windows = m_ampliada[fini_win:ffin_win+1, cini_win:cfin_win+1]
            if es_maximo_centro(windows) and centro_matriz(windows) >= threshold:
                c = columna - win_size
                f = fila - win_size
                print ("(u0:" + str(dx[f, c]) + ", u1: " + str(dy[f, c]) + ")" )
                maximos.append(cv2.KeyPoint(c*(2**nivel_piramide), f*(2**nivel_piramide), _size = (nivel_piramide+1)*block_size, _angle=orientacion(dx[f, c], dy[f, c])))
    return maximos


def get_puntos_harris(imagen, dx, dy, block_size = 5, ksize = 3, threshold = 10, win_size = 5, nivel_piramide = 1):
    matrizH = cv2.cornerEigenValsAndVecs(imagen, block_size, ksize)
    matrizH = cv2.split(matrizH)
    lambda1 = matrizH[0]
    lambda2 = matrizH[1]

    harris = operador_harris(lambda1, lambda2)
    maximos = supresion_no_maximos(imagen, dx, dy, harris, win_size, nivel_piramide, block_size, threshold)
    return maximos


def get_matches_bf_cc(imagen1, imagen2, n = 100, flag = 2):
    # Se obtienen los keypoints y los descriptores de las dos imágenes
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(imagen1, None)
    kpts2, desc2 = akaze.detectAndCompute(imagen2, None)
    bf = cv2.BFMatcher(crossCheck = True)
    matches = bf.match(desc1, desc2)
    ''' SI NO SE PUEDE USAR CROSSCHECK = TRUE
    matches = []
    matches1to2 = bf.match(desc1, desc2)
    matches2to1 = bf.match(desc2, desc1)


    queryIdx = []
    trainIdx = []


    for i in range(len(matches1to2)):
        queryIdx.append(matches1to2[i].queryIdx)
    for i in range(len(matches2to1)):
        trainIdx.append(matches2to1[i].queryIdx)

    if len(matches1to2) < len(matches2to1):
        for i in range(len(matches1to2)):
            if matches1to2[i].queryIdx in trainIdx:
                matches.append(matches1to2[i])
    else:
        for i in range(len(matches2to1)):
            if matches2to1[i].queryIdx in queryIdx:
                matches.append(matches2to1[i])
    '''

    #matches = sorted(matches, key = lambda x:x.distance)[0:n]
    matches = random.sample(matches, n)
    return cv2.drawMatches(imagen1, kpts1, imagen2, kpts2, matches, None, flags = flag)


def get_matches_knn(imagen1, imagen2, k = 2, ratio = 0.8, n = 100, flag = 2):
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(imagen1, None)
    kpts2, desc2 = akaze.detectAndCompute(imagen2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k)
    good = []
    for p1, p2 in matches:
        if p1.distance < ratio*p2.distance:
            good.append([p1])
    #matches = sorted(good, key = lambda x:x[0].distance)
    matches = random.sample(good, n)
    return cv2.drawMatchesKnn(imagen1, kpts1, imagen2, kpts2, matches, None, flags = flag)

def apartado1AB():
    # Parametros
    #im = lee_imagen('imagenes/yosemite/Yosemite1.jpg')
    im = lee_imagen('imagenes/Tablero1.jpg')
    imagen = np.copy(im)

    #imagen = imagen.astype(np.uint8)
    nivel_piramide = 4

    # Realización
    piramide = piramide_gaussiana(imagen, nivel = nivel_piramide)
    puntos_harris = []
    puntos = []

    dx, dy = derivadas(imagen, sigma = 4.5)
    p_dx = piramide_gaussiana(dx, nivel = nivel_piramide)
    p_dy = piramide_gaussiana(dy, nivel = nivel_piramide)
    imagen = imagen.astype(np.uint8)

    # Se recorre cada imagen y se le calculan los puntos Harris
    for i, img in enumerate(piramide):
        puntos = get_puntos_harris(img, p_dx[i], p_dy[i], block_size = 5, ksize = 3, threshold = 10, win_size = 5, nivel_piramide = i)
        puntos_harris += puntos
        print("Numero de puntos harris encontrados en la octava " + str(i+1) + ": " + str(len(puntos)))
        imagen_key = cv2.drawKeypoints(imagen, puntos, np.array([]), color = (0,0,255), flags = 4)
        #imagen_key = pinta_circulos(imagen, puntos_harris)
        pinta_imagen(imagen_key)

    print("Numero de puntos harris encontrados en total: " + str(len(puntos_harris)))
     # Se añade la imagen a la lista de imágenes
    imagen_key = cv2.drawKeypoints(imagen, puntos_harris, np.array([]), color = (0,0,255), flags = 4)
    #imagen_key = pinta_circulos(imagen, puntos_harris)
    pinta_imagen(imagen_key)

def apartado2A():
    # Parametros
    imagen1 = lee_imagen('imagenes/yosemite/Yosemite1.jpg')
    imagen2 = lee_imagen('imagenes/yosemite/Yosemite2.jpg')

    # Realización
    imagenes = []
    imagenes.append(get_matches_bf_cc(imagen1, imagen2))
    show_images(imagenes, cols = 1)

def apartado2B():
    # Parametros
    imagen1 = lee_imagen('imagenes/yosemite/Yosemite1.jpg')
    imagen2 = lee_imagen('imagenes/yosemite/Yosemite2.jpg')

    # Realización
    imagenes = []
    imagenes.append(get_matches_knn(imagen1, imagen2))
    show_images(imagenes, cols = 1)

def ej1():
    print("Ejercicio 1")
    apartado1AB()
    #input("Pulsa enter para continuar")


def ej2():
    print("Ejercicio 2")
    apartado2A()
    input("Pulsa enter para continuar")
    apartado2B()

def main():
    ej1()
    #ej2()



if __name__ == "__main__":
    main()
