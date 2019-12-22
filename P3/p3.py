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


''' ************************** FUNCIONES P1 ******************************** '''

# Funcion para leer las imagenes de la P1
def lee_imagen(file_name, flag):
    if flag == 0:
        imagen = cv2.imread(file_name, flag)
        imagen = imagen.astype(np.float32)
        return imagen
    elif flag == 1:
        imagen = cv2.imread(file_name, flag)
        imagen = imagen.astype(np.float32)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        return imagen
    else:
        sys.exit("Imagen no valida")

# Funcion para pintar las imagenes de la P1
def pinta_imagen(imagen, titulo = "Titulo"):
    imagen = imagen.astype(np.uint8)
    plt.figure(0).canvas.set_window_title(titulo)
    plt.imshow(imagen, cmap = "gray")
    plt.show()

# Funcion para pintar la secuencia de iamgenes de la P1
def pinta_lista_imagenes(secuencia_imagenes):
    fig = plt.figure(0)
    for i in range(len(secuencia_imagenes)):
        secuencia_imagenes[i] = secuencia_imagenes[i].astype(np.uint8)
        plt.subplot(1, 1, i+1)
        plt.imshow(secuencia_imagenes[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Calculo de la convolución una máscara arbitraria de la P1.
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

# Calculo del alisado gassiano una máscara de dimension 2 de la P1.
def gaussian_blur(imagen, kx = 0, ky = 0, sigmax = 0, sigmay = 0, border = cv2.BORDER_DEFAULT):
    if kx == 0:
        kx = int(6*sigmax) + 1
    if ky == 0:
        ky = int(6*sigmay) + 1
    kernelX = cv2.getGaussianKernel(kx, sigmax)
    kernelY = cv2.getGaussianKernel(ky, sigmay)
    imagen = convolucion(imagen, kernelX, kernelY, border)
    return imagen

# Representación en pirámide Gaussiana de una imagen de la P1.
def piramide_gaussiana(imagen, nivel = 4, border = cv2.BORDER_DEFAULT):
    p = [imagen]
    for n in range(nivel-1):
        p.append(cv2.pyrDown(p[-1], borderType = border))
    return p

# Cálculo de las derivadas de una imagen de la P1.
def derivadas(imagen, ksize = 3, sigma = 0, border = cv2.BORDER_DEFAULT):
    imagen = gaussian_blur(imagen, kx = ksize, ky = ksize, sigmax = sigma, sigmay = sigma)
    kx = cv2.getDerivKernels(1, 0, ksize)
    ky = cv2.getDerivKernels(0, 1, ksize)
    dx = convolucion(imagen, kx[0], kx[1], border = border)
    dy = convolucion(imagen, ky[0], ky[1], border = border)
    return dx, dy

''' ************************************************************************ '''

''' ************************** FUNCIONES AUXILIARES ************************ '''

# Operador de Harris
def operador_harris(lambda1, lambda2):
    '''
    Det(matriz)/Traza(matriz)
    Parametros:
        - lambda1: matriz de los primeros valores propios.
        - lambda2: marriz de los segundos valores propios.
    '''
    # Rellenamos la matriz de ceros
    res = np.zeros(lambda1.shape)
    for i in range(lambda1.shape[0]):
        for j in range (lambda1.shape[1]):
            if lambda1[i][j]+lambda2[i][j] == 0:
                res[i][j] = 0
            else:
                #Det(matriz)/Traza(matriz) = lambda1*lambda2/(lambda1+lambda2)
                res[i][j] = lambda1[i][j]*lambda2[i][j]/(lambda1[i][j]+lambda2[i][j])
    return res


# Calcula el centro de la matriz
def centro_matriz(m):
    '''
    Calcula el centro de una matriz cualquiera (obviamente para existir un solo pixel
    central, la matriz debe ser cuadrada de dimensiones impares)
    Parametros:
        - m: matriz
    '''
    # Se calcula el centro de la matriz como dimensiones divididas entre dos
    return m[int(m.shape[0]/2), int(m.shape[1]/2)]

# Indica si el centro de la matriz es el máximo de ésta
def es_maximo_centro(m):
    '''
    Indica si centro de una matriz (obviamente para existir un solo pixel
    central, la matriz debe ser cuadrada de dimensiones impares) es el valor
    máximo de ésta
    Parametros:
        - m: matriz
    '''
    return centro_matriz(m) == np.max(m)

# Calcula la orientacion del vector pasado
def orientacion(u1, u2):
    '''
    Calcula el ángulo en grados entre dos vectores pasados sabiendo que
    (cos(theta),sen(theta))=u/|u|, u=(u1,u2)
    Así a arcotangente de sen(theta)/cos(theta) = theta
    Parametros:
        - u1: cos()
        - u2: sen()
    '''
    norma = math.sqrt(u1*u1+u2*u2)
    if (norma != 0 ):
        # Normalizamos el vector por coordenadas
        u1 = u1 / norma
        u2 = u2 / norma
        # Calculamos la arcotangente en grados
        theta = math.atan2(u2, u1) * 180 / math.pi
        # La trasladamos a un valor entre 0 y 360
        theta += 180
    else:
        theta = 0
    # Devolvemos en grados
    return theta


# Hacemos una supresion de no-máximos de la matriz m y devolvemos los puntos de Harris de la imagen
def supresion_no_maximos(dx, dy, m, win_size, nivel_piramide, block_size, threshold):
    '''
    Calcula los puntos de Harris de la imagen, haciendo una supresion de no-maximos
    por matrices pequeñas (ventanas) de la matriz de Harris de la imagen.
    Parametros:
        - dx: derivada primera de la imagen en el eje x
        - dy: derivada segunda de la imagen en el eje y
        - m: matriz de Harris de la imagen
        - win_size: dimensiones de la ventana = win_size*2+1
        - nivel_piramide: nivel de la piramide gaussiana en que nos encontramos
        - block_size: tamaño del vecindario sobre el que se calcula la matriz de derivadas sobre cada píxel
        - threshold: umbral que sirve como supresion de no-maximos
    '''
    # Calculamos la matriz ampliada donde pasará la ventana sin tener en cuenta las restricciones
    # para ello, copiamos la matriz de Harris en el centro de la nueva matriz y en los laterales
    # la rellenamos de ceros
    m_ampliada = np.ndarray(shape=(m.shape[0]+2*win_size, m.shape[1]+2*win_size))
    m_ampliada[:, :] = 0
    m_ampliada[win_size:m.shape[0]+win_size, win_size:m.shape[1]+win_size] = m.copy()

    maximos = []
    for fila in range(win_size, m.shape[0]+win_size):
        for columna in range(win_size, m.shape[1]+win_size):
            # Para cada punto de la matriz de Harris calculamos la ventana
            fini_win = fila - win_size
            ffin_win = fila + win_size
            cini_win = columna - win_size
            cfin_win = columna + win_size
            windows = m_ampliada[fini_win:ffin_win+1, cini_win:cfin_win+1]
            # Si la ventana tiene como máximo su centro y es superior al umbral
            if es_maximo_centro(windows) and centro_matriz(windows) >= threshold:
                c = columna - win_size
                f = fila - win_size
                # Incluimos dicho pixel como Keypoint con:
                # Coordenadas = (columna * 2^nivel de la piramide, fila * 2^nivel_piramide)
                # Escala = Nivel de la piramide(empezando en 1) * block_size
                # Orientacion = angulo formado por las derivadas de la imagen en dicha fila y columna
                maximos.append(cv2.KeyPoint(c*(2**nivel_piramide), f*(2**nivel_piramide), _size = (nivel_piramide+1)*block_size, _angle=orientacion(dx[f, c], dy[f, c])))
    return maximos

# Calcula los puntos de Harris de la imagen calculando el operador de Harris de la imagen y haciendo la supresion_no_maximos
def get_puntos_harris(imagen, dx, dy, block_size = 5, ksize = 3, threshold = 10, win_size = 5, nivel_piramide = 1):
    '''
    Calcula los puntos de Harris de la imagen, calculando el operador de Harris de la imagen
    y haciendo una supresion de no-maximos por matrices pequeñas (ventanas) de esta.
    Parametros:
        - imagen: imagen original de la que extraer los puntos de Harris.
        - dx: derivada primera de la imagen en el eje x
        - dy: derivada segunda de la imagen en el eje y
        - block_size: tamaño del vecindario sobre el que se calcula la matriz de derivadas sobre cada píxel
        - ksize: tamaño de la máscara de la convolucion
        - win_size: dimensiones de la ventana = win_size*2+1
        - nivel_piramide: nivel de la piramide gaussiana en que nos encontramos
    '''
    # Calculamos los valores y vectores propios de la imagen
    matrizH = cv2.cornerEigenValsAndVecs(imagen, block_size, ksize)
    # Nos quedamos con los valores propios de ésta
    matrizH = cv2.split(matrizH)
    lambda1 = matrizH[0]
    lambda2 = matrizH[1]
    # Calculamos la matriz de Harris aplicando el operador de Harris a cada pixel
    harris = operador_harris(lambda1, lambda2)
    # Realizamos una supresión de no-máximos de la matriz de Harris y calculamos los puntos de Harris
    maximos = supresion_no_maximos(dx, dy, harris, win_size, nivel_piramide, block_size, threshold)
    return maximos

# Ajusta los puntos Harris de la imagen
def ajusta_puntos(imagen, puntos):
    '''
    Calcula 3 puntos ajustados a la posicion correcta
    Parametros:
        - imagen: imagen
        - puntos: puntos Harris obtenidos anteriormente
    '''
    puntos_ajustados = []
    j = 0
    lista_zoom = []
    # Encontramos los puntos ajustados
    p = np.array([punto.pt for punto in puntos], dtype = np.uint32)
    p_ajustados = p.reshape(len(puntos), 1, 2).astype(np.float32)
    cv2.cornerSubPix(imagen, p_ajustados, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    # Buscamos tres puntos ajustados que no coincidan con los originales
    while j < 3:
        numero_aleatorio = random.randint(0, len(p)-1)
        if (p[numero_aleatorio] != p_ajustados[numero_aleatorio][0]).any():
            j += 1
            puntos_ajustados.append(numero_aleatorio)

    # Cambiamos la imagen a color
    img = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB).astype(np.float32)

    # Para cada punto adjustado hacer...
    for i in puntos_ajustados:
        columna_adjustado, fila_adjustado = p_ajustados[i][0]
        columna, fila = p[i]
        # Definir una ventana de tamaño 10x10
        m_ampliada = np.ndarray(shape=(img.shape[0]+2*5, img.shape[1]+2*5, 3))
        m_ampliada[:, :] = 0
        m_ampliada[5:img.shape[0]+5, 5:img.shape[1]+5] = img.copy()
        fini_win = fila - 5
        ffin_win = fila + 5
        cini_win = columna - 5
        cfin_win = columna + 5
        ventana = m_ampliada[fini_win:ffin_win+1, cini_win:cfin_win+1]
        # Hacemos zoom de x5
        ventana = cv2.resize(ventana, None, fx = 5, fy = 5)
        # Señalamos de color verde el punto adjustado
        ventana = cv2.circle(ventana, (int(5*(5+columna_adjustado-columna)+1), int(5*(5+fila_adjustado-fila)+1)), 3, (0, 255, 0))
        # Señalamos de color azul el punto adjustado
        ventana = cv2.circle(ventana, (5*5+1, 5*5+1), 3, (0, 0, 255))
        # Lo guardamos
        lista_zoom.append(ventana)
    return lista_zoom


# Calcula los matches entre dos imagenes con el criterio correspondencia de Fuerza Bruta
# y con los descriptores AKAZE de opencv
def get_matches_fuerza_bruta(imagen1, imagen2, n = 100, flag = 2, pintar = True):
    '''
    Calcula los matches entre dos imagenes con el criterio correspondencia de Fuerza Bruta
    y con los descriptores AKAZE de opencv
    Parametros:
        - imagen1: primera imagen.
        - imagen2: segunda imagen.
        - n: numero de matches finales resultado
        - flag: Banderas que configuran las características de dibujo.
        - pintar: Indica si queremos pintar el resultado u obtener los datos (si esta a false)
    '''
    # Detecta los keyPoints y calcula los descriptores de las imagenes.
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(imagen1, None)
    kpts2, desc2 = akaze.detectAndCompute(imagen2, None)
    # Brute-force descriptor matcher.
    bf = cv2.BFMatcher(crossCheck = True)
    # Calculamos los matches entre la imagen 1 y la imagen 2
    matches = bf.match(desc1, desc2)
    # Nos quedamos con n aleatorios
    matches = random.sample(matches, n)
    if pintar:
        # Los pintamos si el usuario lo pide
        return cv2.drawMatches(imagen1, kpts1, imagen2, kpts2, matches, None, flags = flag)
    # Los devolvemos junto con los keyPoints y los descriptores
    return kpts1, desc1, kpts2, desc2, matches


# Calcula los matches entre dos imagenes con el criterio correspondencia de Lowe-Average-2NN
# y con los descriptores AKAZE de opencv
def get_matches_lowe_average_2nn(imagen1, imagen2, ratio = 0.8, n = 100, flag = 2, pintar = True):
    '''
    Calcula los matches entre dos imagenes con el criterio correspondencia de Lowe-Average-2NN
    y con los descriptores AKAZE de opencv
    Parametros:
        - imagen1: primera imagen.
        - imagen2: segunda imagen.
        - ratio: ratio minimo para decidir cual es el match correcto
        - n: numero de matches finales resultado
        - flag: Banderas que configuran las características de dibujo.
        - pintar: Indica si queremos pintar el resultado u obtener los datos (si esta a false)
    '''
    # Detecta los keyPoints y calcula los descriptores de las imagenes.
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(imagen1, None)
    kpts2, desc2 = akaze.detectAndCompute(imagen2, None)
    # Brute-force descriptor matcher.
    bf = cv2.BFMatcher()
    # Encuentra los 2 mejores matches para cada descriptor.
    matches = bf.knnMatch(desc1, desc2, 2)
    matches_correctos = []
    for m1, m2 in matches:
        # Si la distancia de los matches es pequeña no se elige el match
        if m1.distance < m2.distance * ratio:
            matches_correctos.append([m1])
    # Nos quedamos con n aleatorios
    matches = random.sample(matches_correctos, n)
    if pintar:
        # Los pintamos si el usuario lo pide
        return cv2.drawMatchesKnn(imagen1, kpts1, imagen2, kpts2, matches, None, flags = flag)

    # Los devolvemos junto con los keyPoints y los descriptores
    return kpts1, desc1, kpts2, desc2, matches


# Calcula la homografia entre dos imagenes
def get_homografia(imagen1, imagen2):
    '''
    Calcula la matriz que define la homografia entre dos imágenes
    Parametros:
        - imagen1: primera imagen.
        - imagen2: segunda imagen.
    '''
    # Calculamos los keyPoints y los matches entre las dos imagenes mediante el método de lowe_average_2nn
    kpts1, desc1, kpts2, desc2, matches = get_matches_lowe_average_2nn(imagen1, imagen2, pintar = False)
    # Ordeno los puntos para el findHomography por queryIdx y por trainIdx
    puntos_origen = np.float32([kpts1[punto[0].queryIdx].pt for punto in matches]).reshape(-1, 1, 2)
    puntos_destino = np.float32([kpts2[punto[0].trainIdx].pt for punto in matches]).reshape(-1, 1, 2)
    # Uso la funcion de opencv pa calcular la homografia findHomography con los parametros  cv2.RANSAC, 1
    homografia , _ = cv2.findHomography(puntos_origen, puntos_destino, cv2.RANSAC, 1)
    return homografia

# Calcula la homografia que lleva la imagen a la izquierda de un mosaico dado por sus dimensiones
def homografia_identidad_izq(imagen, ancho_mosaico, alto_mosaico):
    '''
    Calcula la matriz que define la homografia que lleva la imagen a la izquierda del mosaico
    dado por sus dimensiones
    Parametros:
        - imagen: imagen original.
        - ancho_mosaico: ancho del mosaico.
        - alto_mosaico: alto del mosaico.
    '''
    # Devolvemos la matriz identidad para llevarla a la izquierda del mosaico
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

# Calcula la homografia que lleva la imagen al centro del mosaico dado por sus dimensiones
def homografia_identidad(imagen, ancho_mosaico, alto_mosaico):
    '''
    Calcula la matriz que define la homografia que lleva la imagen al centro del mosaico
    dado por sus dimensiones
    Parametros:
        - imagen: imagen original.
        - ancho_mosaico: ancho del mosaico.
        - alto_mosaico: alto del mosaico.
    '''
    # Calculamos las traslaciones tx y ty
    tx = ancho_mosaico/2 - imagen.shape[0]/2
    ty = alto_mosaico/2 - imagen.shape[1]/2
    # Devolvemos la matriz identidad sumada las traslaciones necesarias para llevarla al centro tx y ty
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)


# Calcula el mosaico resultante pasadas 2 imágenes
def get_mosaico2(imagen1, imagen2):
    '''
    Calcula el mosaico resultante pasadas N imágenes.
    Parametros:
        - args: Lista de imagen de número aleatorio
    '''
    # Definimos las dimensiones del mosaico
    alto_mosaico = imagen1.shape[0]
    ancho_mosaico = imagen1.shape[1] + imagen2.shape[1]
    # Homografia que lleva la imagen a la izquierda mosaico
    homografia_id = homografia_identidad_izq(imagen1, ancho_mosaico, alto_mosaico)
    # Aplicamos la homografia a la imagen con el parametro BORDER_TRANSPARENT
    imagen = cv2.warpPerspective(imagen1, homografia_id, (ancho_mosaico, alto_mosaico), borderMode=cv2.BORDER_TRANSPARENT)
    # Conseguimos la homografia que lleva la imagen1 a la 2
    homografia = get_homografia(imagen2, imagen1)
    # Componemos con la homografía identidad para llevarla al mosaico
    homografia = np.dot(homografia_id, homografia)
    # Aplicamos la homografia a la imagen con el parametro BORDER_TRANSPARENT
    imagen = cv2.warpPerspective(imagen2, homografia, (ancho_mosaico, alto_mosaico), dst=imagen, borderMode=cv2.BORDER_TRANSPARENT)
    return imagen

# Calcula el mosaico resultante pasadas N imágenes
def get_mosaicoN(*args):
    '''
    Calcula el mosaico resultante pasadas N imágenes.
    Parametros:
        - args: Lista de imagen de número aleatorio
    '''
    # Se obtiene la imagen central que estará en el centro del mosaico
    imagen_central =  args[int(len(args)/2)]
    # Se calculan las dimensiones del mosaico
    alto_mosaico = args[0].shape[0]+args[0].shape[0]
    #ancho_mosaico = sum([imagen.shape[1] for imagen in args])
    ancho_mosaico = args[0].shape[1]*3
    # Se calcula la homografia que lleva la imagen central al centro del mosaico
    homografia_id = homografia_identidad(imagen_central, ancho_mosaico, alto_mosaico)
    # Aplicamos la homografia a la imagen con el parametro BORDER_TRANSPARENT
    imagen = cv2.warpPerspective(imagen_central, homografia_id, (ancho_mosaico, alto_mosaico), borderMode=cv2.BORDER_TRANSPARENT)
    # Rellenamos la lista donde guardaremos las homografías obtenidas.
    lista_homografias = []
    for i in range(len(args)):
        lista_homografias.append(None)
    # Incluimos la homografia de la imagen central al centro del mosaico
    lista_homografias[int(len(args)/2)] = homografia_id

    # Para cada imagen de la derecha hacer...
    for i in range(int(len(args)/2)+1, len(args)):
        # Obtener la homografia entre la imagen y la anterior
        homografia_aux = get_homografia(args[i], args[i-1])
        # Componemos con la homografía anterior para llevarla al mosaico
        homografia_aux = np.dot(lista_homografias[i-1], homografia_aux)
        # Aplicamos la homografia a la imagen con el parametro BORDER_TRANSPARENT
        imagen = cv2.warpPerspective(args[i], homografia_aux, (ancho_mosaico, alto_mosaico), dst=imagen, borderMode=cv2.BORDER_TRANSPARENT)
        # Incluimos la homografía obtenida en la lista de homografías
        lista_homografias[i] = homografia_aux

    # Para cada imagen de la izq hacer...
    for i in range(0, int(len(args)/2))[::-1]:
        # Obtener la homografia entre la imagen y la siguiente
        homografia_aux = get_homografia(args[i], args[i+1])
        # Componemos con la homografía siguiente para llevarla al mosaico
        homografia_aux = np.dot(lista_homografias[i+1], homografia_aux)
        # Aplicamos la homografia a la imagen con el parametro BORDER_TRANSPARENT
        imagen = cv2.warpPerspective(args[i], homografia_aux, (ancho_mosaico, alto_mosaico), dst=imagen, borderMode=cv2.BORDER_TRANSPARENT)
        # Incluimos la homografía obtenida en la lista de homografías
        lista_homografias[i] = homografia_aux

    return imagen

def apartado1AB():
    # PARAMETROS:
    im = lee_imagen('imagenes/yosemite/Yosemite1.jpg', 0)
    #im = lee_imagen('imagenes/Tablero1.jpg', 0)
    imagen = np.copy(im)
    nivel_piramide = 4

    # REALIZACION:
    # Obtención de la piramide gaussiana
    piramide = piramide_gaussiana(imagen, nivel = nivel_piramide)
    puntos_harris = []
    puntos = []

    # Obtención de la piramide gaussiana de las derivadas con sigma = 4.5
    dx, dy = derivadas(imagen, sigma = 4.5)
    p_dx = piramide_gaussiana(dx, nivel = nivel_piramide)
    p_dy = piramide_gaussiana(dy, nivel = nivel_piramide)

    # Ya solo trabajaremos con la imagen original para pintarla, por eso la pasamos a enteros
    imagen = imagen.astype(np.uint8)

    for i, img in enumerate(piramide):
        # Para cada nivel de la piramide se obtienen los puntos de harris
        puntos = get_puntos_harris(img, p_dx[i], p_dy[i], block_size = 5, ksize = 5, threshold = 10, win_size = 3, nivel_piramide = i)
        puntos_harris += puntos
        print("Numero de puntos Harris encontrados en la octava " + str(i+1) + ": " + str(len(puntos)))
        # Dibujamos en la imagen los puntos obtenidos
        imagen_key = cv2.drawKeypoints(imagen, puntos, np.array([]), color = (0,0,255), flags = 4)
        # Mostramos el resultado
        pinta_imagen(imagen_key)

    print("Numero de puntos Harris encontrados en total: " + str(len(puntos_harris)))
     # Se añade la imagen a la lista de imágenes
    imagen_key = cv2.drawKeypoints(imagen, puntos_harris, np.array([]), color = (0,0,255), flags = 4)
    pinta_imagen(imagen_key)

def apartado1D():
    # PARAMETROS:
    im = lee_imagen('imagenes/yosemite/Yosemite1.jpg', 0)
    #im = lee_imagen('imagenes/Tablero1.jpg', 0)
    imagen = np.copy(im)
    nivel_piramide = 4

    # REALIZACION:
    # Obtención de la piramide gaussiana
    piramide = piramide_gaussiana(imagen, nivel = nivel_piramide)
    puntos_harris = []
    puntos = []

    # Obtención de la piramide gaussiana de las derivadas con sigma = 4.5
    dx, dy = derivadas(imagen, sigma = 4.5)
    p_dx = piramide_gaussiana(dx, nivel = nivel_piramide)
    p_dy = piramide_gaussiana(dy, nivel = nivel_piramide)

    # Ya solo trabajaremos con la imagen original para pintarla, por eso la pasamos a enteros
    imagen = imagen.astype(np.uint8)

    for i, img in enumerate(piramide):
        # Para cada nivel de la piramide se obtienen los puntos de harris
        puntos = get_puntos_harris(img, p_dx[i], p_dy[i], block_size = 5, ksize = 5, threshold = 10, win_size = 3, nivel_piramide = i)
        puntos_harris += puntos
     # Se añade la imagen a la lista de imágenes
    subpix1 = ajusta_puntos(imagen, puntos_harris)
    for i, img in enumerate(subpix1):
        pinta_imagen(img)


def apartado2A():
    # Parametros
    imagen1 = lee_imagen('imagenes/yosemite/Yosemite1.jpg', 1)
    imagen2 = lee_imagen('imagenes/yosemite/Yosemite2.jpg', 1)

    # Realización
    imagenes = []
    imagenes.append(get_matches_fuerza_bruta(imagen1, imagen2, pintar = True))
    pinta_lista_imagenes(imagenes)

def apartado2B():
    # Parametros
    imagen1 = lee_imagen('imagenes/yosemite/Yosemite1.jpg', 1)
    imagen2 = lee_imagen('imagenes/yosemite/Yosemite2.jpg', 1)

    # Realización
    imagenes = []
    imagenes.append(get_matches_lowe_average_2nn(imagen1, imagen2, pintar = True))
    pinta_lista_imagenes(imagenes)

def apartado3():
    mosaico = []
    imagen1 = lee_imagen('imagenes/yosemite/Yosemite1.jpg', 1)
    imagen2 = lee_imagen('imagenes/yosemite/Yosemite2.jpg', 1)
    mosaico.append(get_mosaico2(imagen1, imagen2))
    pinta_lista_imagenes(mosaico)

def apartado4():
    mosaico1 = []
    mosaico2 = []
    mosaico3 = []
    imagen1 = lee_imagen('imagenes/yosemite_full/yosemite1.jpg', 1)
    imagen2 = lee_imagen('imagenes/yosemite_full/yosemite2.jpg', 1)
    imagen3 = lee_imagen('imagenes/yosemite_full/yosemite3.jpg', 1)
    imagen4 = lee_imagen('imagenes/yosemite_full/yosemite4.jpg', 1)
    imagen5 = lee_imagen('imagenes/yosemite_full/yosemite5.jpg', 1)
    imagen6 = lee_imagen('imagenes/yosemite_full/yosemite6.jpg', 1)
    imagen7 = lee_imagen('imagenes/yosemite_full/yosemite7.jpg', 1)
    mosaico1.append(get_mosaicoN(imagen1, imagen2, imagen3, imagen4))
    mosaico2.append(get_mosaicoN(imagen5, imagen6, imagen7))
    pinta_lista_imagenes(mosaico1)
    pinta_lista_imagenes(mosaico2)

    etsiit1 = lee_imagen('imagenes/mosaico-1/mosaico002.jpg', 1)
    etsiit2 = lee_imagen('imagenes/mosaico-1/mosaico003.jpg', 1)
    etsiit3 = lee_imagen('imagenes/mosaico-1/mosaico004.jpg', 1)
    etsiit4 = lee_imagen('imagenes/mosaico-1/mosaico005.jpg', 1)
    etsiit5 = lee_imagen('imagenes/mosaico-1/mosaico006.jpg', 1)
    etsiit6 = lee_imagen('imagenes/mosaico-1/mosaico007.jpg', 1)
    etsiit7 = lee_imagen('imagenes/mosaico-1/mosaico008.jpg', 1)
    etsiit8 = lee_imagen('imagenes/mosaico-1/mosaico009.jpg', 1)
    etsiit9 = lee_imagen('imagenes/mosaico-1/mosaico010.jpg', 1)
    etsiit10 = lee_imagen('imagenes/mosaico-1/mosaico011.jpg', 1)
    mosaico3.append(get_mosaicoN(etsiit1, etsiit2, etsiit3, etsiit4, etsiit5, etsiit6, etsiit7, etsiit8, etsiit9, etsiit10))
    pinta_lista_imagenes(mosaico3)



def ej1():
    print("Ejercicio 1\n")
    print("\tApartados A-C:\n")
    apartado1AB()
    input("Pulsa enter para continuar\n")
    print("\tApartado D:\n")
    apartado1D()
    input("Pulsa enter para continuar\n")

def ej2():
    print("Ejercicio 2")
    print("\tApartado A:\n")
    apartado2A()
    input("Pulsa enter para continuar")
    print("\tApartado B:\n")
    apartado2B()
    input("Pulsa enter para continuar")

def ej3():
    print("Ejercicio 3")
    apartado3()
    input("Pulsa enter para continuar")

def ej4():
    print("Ejercicio 4")
    apartado4()
    input("Pulsa enter para continuar")


def main():
    ej1()
    ej2()
    ej3()
    ej4()



if __name__ == "__main__":
    main()
