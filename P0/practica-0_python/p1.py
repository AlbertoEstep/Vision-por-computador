# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

# Funcion para leer las imagenes
def lee_imagen(file_name, flag_color):
    # flag_color 0 = Gris
    # flag_color 1 = Color
    if flag_color == 0 or flag_color == 1:
        return cv2.imread(file_name, flag_color)
    else:
        sys.exit('Imagen no valida')

# Funcion para pintar las imagenes
def pinta_imagen_cv2(imagen, titulo):
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.imshow(titulo, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pinta_imagen(imagen, titulo):
    normalizar_imagen(imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    plt.figure(0).canvas.set_window_title(titulo)
    plt.imshow(imagen)
    plt.show()


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

# Funcion para pintar varias imagenes a la vez
def pintar_multiples_imagenes(secuencia_imagenes, posicion):
    # Primero igualamos el tamaño de todas las imagenes
    secuencia_imagenes = igualTam(secuencia_imagenes)

    for i in range(len(secuencia_imagenes)):
        # Si la imagen es monobanda (gris) se transforma en imagen tribanda (BGR)
        if (np.size(np.shape(secuencia_imagenes[i])) == 2):
            secuencia_imagenes[i] = cv2.cvtColor(secuencia_imagenes[i], cv2.COLOR_GRAY2BGR)

        # Juntamos todas las imagenes en una dando libertad al parametro posicion{0 = vertical, 1 = horizontal}
        if (i == 0):
            imagenes_juntas = secuencia_imagenes[i]
        else:
            imagenes_juntas = np.concatenate((imagenes_juntas, secuencia_imagenes[i]), axis = posicion)

    pinta_imagen(imagenes_juntas, 'Pintar varias imagenes juntas')

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


# Función que modifica el color de los pixeles de una imagen
def modificar_color_pixel(imagen, pixeles, tipo_imagen, color):
    if (tipo_imagen == 0 or tipo_imagen == 1):
        img = lee_imagen(imagen, tipo_imagen)
        # Si la imagen es monobanda (gris) se transforma en imagen tribanda (BGR)
        if (tipo_imagen == 0):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        y, x, z = np.shape(img)

        # Para cada pixel de la lista lo pintamos del color pedido
        for p in pixeles:
            if p[0] < y and p[0] >= 0 and p[1] < x and p[1] >= 0:
                img[p[1]][p[0]] = color
            else:
                sys.exit('Pixel no valido')
        pinta_imagen(img, 'Pixeles modificados')
    else:
        sys.exit('Imagen no valida')



# Funcion para pintar varias imagenes a la vez con pyplot
def pintar_multiples_imagenes_pyplot(secuencia_imagenes, tituloImagen, filas, columnas, nombre):
    # Primero igualamos el tamaño de todas las imagenes
    secuencia_imagenes = igualTam(secuencia_imagenes)
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

def main():
    # Ejercicio 1 y Ejercicio 2
    print("------ EJERCICIOS 1 y 2 ------")
    # Se lee la Imagen en Gris
    pinta_imagen(lee_imagen('images/orapple.jpg', 0), 'Imagen en gris')
    # Se lee la Imagen en Color
    pinta_imagen(lee_imagen('images/orapple.jpg', 1), 'Imagen en color')

    """FALTA NORMALIZAR"""

    # Ejercicio 3
    print("------ EJERCICIO 3 ------")
    pintar_multiples_imagenes([lee_imagen('images/orapple.jpg', 0), lee_imagen('images/orapple.jpg', 1), lee_imagen('images/messi.jpg', 1)], 1)

    # Ejercicio 4
    print("------ EJERCICIO 4 ------")
    modificar_color_pixel('images/orapple.jpg', [(x,y) for y in range (200) for x in range (100)], 1, (0,0,255))

    # Ejercicio 5
    print("------ EJERCICIO 5 ------")
    pintar_multiples_imagenes_pyplot([lee_imagen('images/orapple.jpg', 0), lee_imagen('images/orapple.jpg', 1), lee_imagen('images/messi.jpg', 1)], ['Orapple', 'Orapple', 'Messi'], 1, 3, 'Imagenes Concatenadas')

if __name__ == "__main__":
    main()
