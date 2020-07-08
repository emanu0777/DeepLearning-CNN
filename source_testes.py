# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:11:11 2020

@author: Resource
"""
import cv2
import tensorflow as tf
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


#importando os dados para teste
X_test = pickle.load(open("X_test.pickle", "rb"))
X_test = np.array(X_test)


#importando as labels 
Atributos  = pd.read_csv("list_attr_celeba_formatado.csv", sep =";", usecols = ["Indice_IMG", "Smiling"])

troca = { 1:1,
         -1:0
            }

Atributos["Smiling"] = Atributos.Smiling.map(troca)


Lables = Atributos["Smiling"]

y_teste = Lables[10000:11000]



model = tf.keras.models.load_model("modelo-treinado.model")



predicao = model.predict(X_test)

#print(confusion_matrix(y_teste, predicao.round()).ravel())
    
#Vn, Fp, Fn, Vp

Vn, Fp, Fn, Vp = confusion_matrix(y_teste, predicao.round()).ravel()

print("Verdadeiros-Negativos", Vn)
print("Falsos-Positivos", Fp)
print("Falsos-Negativos", Fn)
print("Verdadeiros-Positivos", Vp)



#for i in range(50):
#    print(int(predicao[i][0]))


