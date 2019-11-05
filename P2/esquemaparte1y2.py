# -*- coding: utf-8 -*-

#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

# A completar: esquema disponible en las diapositivas

import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.utils as np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# Importar el optimizador a usar
from keras.optimizers import SGD
# Importar el conjunto de datos
from keras.datasets import cifar100

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# A completar: función disponible en las diapositivas


def cargarImagenes():
    # Cargamos Cifar100. Cada imagen tiene tamaño
    # (32, 32, 3). Nos vamos a quedar con las
    # imágenes de 25 de las clases.

    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape(train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    # Transformamos los vectores de clases en matrices.
    # Cada componente se convierte en un vector de ceros
    # con un uno en la componente correspondiente a la
    # clase a la que pertenece la imagen. Este paso es
    # necesario para la clasificación multiclase en keras.

    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)

    return [x_train, y_train], [x_test, y_test]


#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# A completar: función disponible en las diapositivas

def calcularAccuracy(labels, preds):
  labels = np.argmax(labels, axis=1)
  preds = np.argmax(preds, axis=1)

  accuracy = sum(labels == preds)/len(labels)

  return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# A completar: función disponible en las diapositivas

def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

    accuracy = hist.history['accuracy']
    val_accuracy = hist.history['val_accuracy']
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.show()


#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

# A completar

img_rows, img_cols = 32, 32
num_classes = 25
batch_size = 128
epochs = 12

(x_train, y_train), (x_test, y_test) = cargarImagenes()

input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(6, kernel_size=(5,5), activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='softmax'))
model.summary()

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# A completar
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

histograma = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

mostrarEvolucion(histograma)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los
# pesos aleatorios con los que empieza la red, para poder reestablecerlos
# después y comparar resultados entre no usar mejoras y sí usarlas.
weights = model.get_weights()

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

# A completar

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

# A completar

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.
