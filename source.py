# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:51:00 2020

@author: Resource
"""

import cv2
import pandas as pd
import os
from random import shuffle
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

TRAIN_DIR  = 'C:/Users\DELL/Documents/ProgramaTreinamento/Imagens para Treinamento_Testes/imagens_train'
TEST_DIR = 'C:/Users\DELL/Documents/ProgramaTreinamento/Imagens para Treinamento_Testes/imagens_test'
IMG_SIZE1 = 45
IMG_SIZE2 = 55
LR = 1e-3



#MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')
MODEL_NAME = 'smiling-{}-{}.model'.format(LR, '2conv-basic')

Atributos  = pd.read_csv("list_attr_celeba_formatado.csv", sep =";", usecols = ["Indice_IMG", "Smiling"])

def label_img(img):
     Atributos  = pd.read_csv("list_attr_celeba_formatado.csv", sep =";", usecols = ["Indice_IMG", "Smiling"])
     i =0;
     j=0
     for indice in Atributos['Indice_IMG']:
         if i < 10000:
             if indice == img:
                 print(indice)
                 for label in Atributos['Smiling']:
                     if (label == 1) and (i==j):
                          return 1
                     elif (label == -1) and (i==j):
                         return 0
                     j= j+1
             i = i+1




training_data = []
teste_data = []
    
def create_train_data():
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE1,IMG_SIZE2))
        training_data.append([img, label])    
 

 
def create_teste_data():
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE1,IMG_SIZE2))
        teste_data.append(img)



create_teste_data()

print("Criando dados do treinamento")
create_train_data()





X = []
y = []

X_test  = []
y_test = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE1, IMG_SIZE2, 1)


for features in teste_data:
    X_test.append(features)


X_test = np.array(X_test).reshape(-1,IMG_SIZE1, IMG_SIZE2, 1)



import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X,pickle_out)    
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y,pickle_out)    
pickle_out.close()


pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test,pickle_out)    
pickle_out.close()
