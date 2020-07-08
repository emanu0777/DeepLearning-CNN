# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:35:39 2020

@author: Resource
"""
import tensorflow as tk
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

print(X.shape[1:])


def AlexNet():
    model = Sequential()
    #1Conv
    model.add(Conv2D(filters=96, input_shape=X.shape[1:], kernel_size=(11,11),strides=(4,4), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),  padding='valid'))
    #2Conv
    model.add(Conv2D(filters=256, kernel_size=(5,5),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #3Conv
    model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    #4Conv
    model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    
    #5Conv
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, y, batch_size=50, validation_split=0.1, epochs=15)

    model.save('model_AlexNet_teste.model')
    
    
AlexNet()