# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:09:43 2020

@author: Resource
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import cv2


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X_test = pickle.load(open("X_test.pickle", "rb"))


X=X/255.0


X = np.array(X)
y = np.array(y)
X_test = np.array(X_test)



model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))  #Conv2d(Numero de filtros na saida da convolucao, tamanho da janela, tamanho da imagem(50,50))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))   # Indica a quantidade de colunas e linhas da minha matriz que vai fazer o pooling


model.add(Conv2D(128, (3,3)))
model.add(Activation("relu"))  # Funcao de Ativacao Relu
model.add(MaxPooling2D(pool_size=(2,2)))
 


model.add(Flatten())  # Transforma os resultados obtidos com o pooling em entradas para a rede neural (Primeiros Neuronios)
model.add(Dense(128))  # Indica que  minha camada mais profunda terá o (N parametro) Neuronios ou nós
model.add(Activation('sigmoid'))

model.add(Dense(64)) 

model.add(Dense(1))
model.add(Activation('sigmoid')) # Funcao de Ativacao Sigmoid


 
model.compile(loss="binary_crossentropy",  # Mede quao preciso o modelo é durante o treinamento 
              optimizer = "adam",       # Algoritmo de Adam
              metrics = ['accuracy'])  # Monitora a fracão de imagens que foram classificas corretamente


model.fit(X,y, batch_size=32, epochs=5, validation_split=0.1) # (imgs_entrada, labels, quantidade dos lotes(imgs) que desejo passar, numero de interações sobre os dados X e y para treinar, validation_split)

model.save('modelo-treinado.model')

#predicao = model.predict([X_test])

#print(predicao)
