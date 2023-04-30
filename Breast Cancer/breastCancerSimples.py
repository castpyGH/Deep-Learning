# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 17:06:41 2023

@author: Castpy
"""

import pandas as pd

#Lendo arquivos de entrada e saída
previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")

#Divindo a base entre treinamento e teste
from sklearn.model_selection import train_test_split
#Essas variaveis recebem a divisão das bases feita automaticamente pelo modulo importado a cima.
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

#Implementando a estrutura da rede neural
import keras
from keras.models import Sequential
#O módulo DENSE é para interligar cada neuronio com todos os subsequentes a sua frente
from keras.layers import Dense

#Criando a rede
classificador = Sequential()
#Criando a primeira camada oculta

#units = QUANTIDADE DE NEURONIOS NA CAMADA OCULTA
#para saber a quantidade correta, use a quantidade de entradas + quantidade de saida / 2
#No nosso caso, temos 30 entradas, 1 saida binária, e dividimos por 2. Terminal -> (30 + 2) / 2 = 15.5
#O resultado da soma acima é 15.5, então podemos usar 16

#activation = função de ativação usada na rede neural
#input_dim = quantidade de entradas
classificador.add(Dense(units = 16, activation= 'relu', kernel_initializer= 'random_uniform', input_dim= 30))


#Camada de saída
classificador.add(Dense(units= 1, activation='sigmoid'))

#compilando a rede neural
#Começando com o otimizador ADAM para a descida do gradiente estocástico, para seu caso ou para outros resultados, use outros.
#Loos function é a forma como vamos fazer o cálculo do erro.
#Nosso caso é de classificação binária, então vamos usar o binary_crossentropy na função loss
classificador.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["binary_accuracy"])


#Realizando o treinamento da rede neural
#A função fit irá realizar o treinamento para aprender "encaixar" seus resultados de saída com os valores corretos
#O parâmetro batch_size indica que a rede vai usar 10 registros para treinamento e só depois disso irá atualizar os pesos.
#EPOCHS é a quantidade de vezes ou épocas que queremos relizar o treinamento, iremos começar com 100.
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)