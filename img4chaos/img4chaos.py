"""
La funcion principal se denomina nnchaos y toma como argumento una serie temporal en forma de array unidimensional. Devuelve todas las coloraciones donde el resultado de la clasificacion es mayor que 0 y su resultado.
"""

# imports

import seaborn as sns
import pandas as pd
import numpy as np
import math as mt
from matplotlib import cm
import random as rd
import re
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
import datetime as dt
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# every color_map in matplotlib

cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'twilight', 'twilight_shifted', 'hsv',
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

def hora(): #esto es una tonteria para dar la hora
    now = dt.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("                                                       ", dt_string)

"""
Funciones para llevar de un array 1-dimensional a una matriz desde la que se obtiene la imagen

"""

def ts_to_mat(ts,g=1): # time series, dimension and delay. # TS to Matrix (Taken)
    w=int(len(ts)/3)
    A=np.array(ts)
    return A[(np.arange(w)*(g+1))+ np.arange(np.max(A.shape[0] - (w-1)*(g+1), 0)).reshape(-1,1)]


def mat_norm(mat): # se normaliza a partir del maximo y minimo a una matriz entre 0 y 1
    mx=mat.max()
    mn=mat.min()
    mat=(mat-mn)/(mx-mn) # entre 0 y 1
    return mat


"""
Funciones para crear las imagenes de los tramos de la serie original y de las muestras
"""



def vimg_original(ts,colmap): # se obtienen las imagenes de la serie original a partir de la serie y una coloracion colmap
    images_original=[] # tramo a tramo, 20 patron original
    l=int(len(ts)/20)
    for i in range(20):
        M=mat_norm(ts_to_mat(ts[(i*l):((i+1)*l)]))
        images_original.append(np.uint8(getattr(cm, str(colmap))(M)*255))
    return images_original


def samples(ts,colmap): # se obtienen las imagenes de las muestras a partir de la st y una coloracion
    sample=[]
    l=int(len(ts)/20)
    for i in range(90): # number of samples
        b=np.random.choice(ts, size=l, replace=False)
        M=mat_norm(ts_to_mat(b))
        sample.append(np.uint8(getattr(cm, str(colmap))(M)*255))
    return sample

def esta_i(i,lista): # es una funcion tonta para saber si esta en la lista, seguro que hay una manera mas directa de hacerlo 
    for j in lista:
        if i==j:
            return True
    return False


def train_test(ts,colmap): # se crea el vector de entrenamiento y de test
    originales=vimg_original(ts,colmap)
    sample=samples(ts,colmap)
    listaaleatoria=np.random.choice(range(20), size=10, replace=False)
    for k in listaaleatoria:
        sample.append(originales[k])
    test=[]
    for i in range(20):
        if not esta_i(i,listaaleatoria):
            test.append(originales[i])
    return sample, test

"""
funcion que, desde una ts y un colmap, entrena el modelo y devuelve el resultado (n epochs)
"""

def nn(ts,colmap,nepochs=120):
    sample, test = train_test(ts,colmap)
    labels=np.zeros(len(sample))
    for i in range(1,11):
        labels[-i]=1
    IS=sample[0].shape
    INIT_LR=1e-3
    epochs = nepochs
    batch_size = 64
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=IS))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5)) 
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])
    valid_X=test
    valid_label=[1,1,1,1,1,1,1,1,1,1]
    sample = np.array(sample).astype('float32')
    valid_X = np.array(valid_X).astype('float32')
    sample = sample / 255.
    valid_X = valid_X / 255.
    labels=to_categorical(labels)
    valid_label=to_categorical(valid_label)
    train_dropout = model.fit(sample, labels, batch_size=batch_size,epochs=epochs,verbose=0)
    test_eval = model.evaluate(valid_X, valid_label, verbose=0)
    return test_eval[1]

"""
funcion que recorre todos los colmap para buscar indicios de caos, funcion principal
"""

def nnchaos(ts,nepochs):
    "Comienza el algoritmo..."
    hora()
    print("----------------------------------")
    count=1
    tot=len(cmaps)
    for i in cmaps:
        print("                                                      cmap "+str(count)+" out of "+str(tot))
        hora()
        count=count+1
        res=nn(ts,i,nepochs)
        if res>0.0001:
            print("      Indicio de Caos con coloracion: "+str(i))
            print("      Valor:                          "+str(res))



