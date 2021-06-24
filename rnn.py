import pandas as pd

import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import streamlit as sl
from sklearn.preprocessing import MinMaxScaler 
from math import sqrt
import tensorflow as tf 
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def implement_model_rnn(lr, ep, batch):

    #la fonction qui implèmente le modèle choisi dans l'interface

    weekly_sales = pd.read_csv("Walmart_Store_sales.csv")
    features = weekly_sales.iloc[:, [0,1,3,4,5,6,7]] #on sépare les données caractéristiques de la variable cible

    #on divise les dates en jours, mois et années pour calculer la differences en jours des dates 
    jours = features['Date'].str.slice(0,2,1) 
    jours_array = np.asarray(jours).astype('float32')

    mois = features['Date'].str.slice(3,5,1)
    mois_array = np.asarray(mois).astype('float32')

    annee = features['Date'].str.slice(6,10,1)
    annee_array = np.asarray(annee).astype('float32')

    #on remplace les valeurs de la colonne Date par la difference en jours
    length_date= len(features['Date'].values)
    jours_diff=np.empty(length_date, dtype='float32')
    jours_diff[0] = 0
    for i in range(1,length_date):
        date_suivant = date(annee_array[i], mois_array[i], jours_array[i])
        date_origine = date(annee_array[0], mois_array[0], jours_array[0])
        delta = date_suivant - date_origine
        jours_diff[i] = delta.days

    features['Date'] = jours_diff

    stdr_target = weekly_sales.iloc[:, [2]]
    min_max = MinMaxScaler(feature_range=(0,1))
    stdr_target = min_max.fit_transform(stdr_target)
    X_train, X_valid, Y_train, Y_valid= train_test_split(features, stdr_target, test_size=0.2, random_state=0 )
    X_train, X_valid= np.array(X_train), np.array(X_valid)
    #print("Input d'apprentissage: ", X_train.shape, "Input de validation: ", X_valid.shape, "Output d'apprentissage: ", Y_train.shape, "Output de validation: ", Y_valid.shape)
    X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid_series = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    X_train_series_array = np.asarray(X_train_series).astype(np.float32)
    X_valid_series_array = np.asarray(X_valid_series).astype(np.float32) 
    Y_train_array = np.asarray(Y_train).astype(np.float32) 
    Y_valid_array = np.asarray(Y_valid).astype(np.float32)
    #print('Train set shape', X_train_series_array.shape)
    #print('Validation set shape', X_valid_series_array.shape)
    #neurons per layer tests
    #[44, 45,  3]
    #(19,12,10)best so far 
    #[ 9,  7, 26] 
    #[5, 5, 2]/([19, 25,  4])


    epochs=int(ep)
    batch_size = int(batch)
    batch_list= (75,100,125,150,175)
    opt = Adam(float(lr))
    #on utilise un moodèle de RNN empilé de 3
    rnn_model =Sequential([
    SimpleRNN(19, return_sequences=True, input_shape=(X_train_series_array.shape[1], X_train_series_array.shape[2])),
    SimpleRNN(12, return_sequences=True),
    SimpleRNN(10, return_sequences=False),
    Dense(1)
    ])
    rnn_model.compile(loss="mse", optimizer=opt)
    for i in range(0,5):
        history = rnn_model.fit(X_train_series_array, Y_train_array, validation_data=(X_valid_series_array, Y_valid_array), epochs=400, verbose=2, batch_size = batch_list[i] , shuffle = False)



    history = rnn_model.fit(X_train_series_array, Y_train_array, validation_data=(X_valid_series_array, Y_valid_array), epochs=epochs, verbose=2, batch_size = batch_size , shuffle = False)

    if(Y_valid[0] != 1220984.94):
        Y_valid=min_max.inverse_transform(Y_valid)
    Y_pred=rnn_model.predict(X_valid_series_array)
    Y_pred = min_max.inverse_transform(Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_valid, Y_pred))
    mean=0
    for i in range(len(weekly_sales)):
        mean += weekly_sales['Weekly_Sales'][i]
    mean = mean/len(weekly_sales)
    rmse= (100*rmse)/mean
    print(rmse)
    #sl.write("L'erreur moyenne de validation: ",rmse)
    err=0
    for i in range(len(Y_pred)):
        err = (Y_pred[i]-Y_valid[i])/Y_valid[i] + err
    err=(err*100)/len(Y_pred)
    print(err)
    sl.write("Source Dataset: Walmart_Store_sales.csv")
    sl.write("Modèle: RNN")
    sl.write("L'erreur de validation: ",err)
    plt.plot(Y_valid)
    plt.plot(Y_pred)
    #on visualise les valeurs réelles et prédites
    sl.set_option('deprecation.showPyplotGlobalUse', False)
    sl.pyplot()
    #0.18725830545659986
    #[2.65662455]


implement_model_rnn(0.001, 300,200)