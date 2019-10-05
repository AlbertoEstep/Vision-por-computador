# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


# El cálculo de la convolución de una imagen con una máscara 2D. Usar una Gaussiana 2D (GaussianBlur)
def gaussian_blur(img, size, sigma, border = cv2.BORDER_DEFAULT):
	return cv2.GaussianBlur(img, size, sigma, border)


# Máscaras 1D dadas por getDerivKernels
def derive_convolution(derivX = 0, derivY = 0, size = 7, normal = True):
	#Ksize = 1, 3, 5, 7
    if ((size == 1) or (size == 3) or (size == 5) or (size == 7)):
        return cv2.getDerivKernels(dx = derivX, dy = derivY, ksize = size, normalize = normal, ktype = cv2.CV_64F)
    #Si el Ksize no es valido se obtiene error
    else:
        print('El tamaño debe ser 1, 3, 5 o 7')
        sys.exit()



def main():



if __name__ == "__main__":
    main()
