import pandas as pd
import io 
import math
import numpy as np
import streamlit as sl
from datetime import date
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.python.data import Dataset

def implement_model_dnn(model, lr, ep, batch):
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
    min_max = MinMaxScaler(feature_range=(100,1000))
    stdr_target = min_max.fit_transform(stdr_target)
    X_train, X_valid, Y_train, Y_valid= train_test_split(features, stdr_target, test_size=0.2, random_state=0 )
    def construct_feature_columns(input_features):
        return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])
    def my_input_fn(feature, target, batch_size=1, num_epochs=None):

        # Convert pandas data into a dict of np arrays.
        feature = {key:np.array(value) for key,value in dict(feature).items()}                                             
    
        # Construct a dataset, and configure batching/repeating.
        ds = Dataset.from_tensor_slices((feature,target)) # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)
        # Return the next batch of data.
        feature, labels = ds.make_one_shot_iterator().get_next()
        return feature, labels
    #neurons per layer tests
    

    learning_rate = float (lr)
    batch_size= int (batch)
    hidden_units=[19,12,10]
    steps=int (ep + '1900')
    my_optimizer = optimizers.Adam(learning_rate)
    dnn_regressor = tf.estimator.DNNRegressor(feature_columns=construct_feature_columns(X_train),hidden_units=hidden_units,optimizer=my_optimizer,activation_fn=tf.nn.swish)
    # Create input functions.
    training_input_fn = lambda: my_input_fn(X_train, Y_train, batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(X_train,Y_train, num_epochs=1)
    predict_validation_input_fn = lambda: my_input_fn(X_valid,Y_valid,num_epochs=1)

    #train the model 
    dnn_regressor.train(input_fn=training_input_fn,steps=steps)

    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

    Y_valid=min_max.inverse_transform(Y_valid)
    validation_predictions=min_max.inverse_transform(validation_predictions.reshape(-1,1))
    validation_root_mean_squared_error = np.sqrt(mean_squared_error(validation_predictions, Y_valid))
    training_root_mean_squared_error = np.sqrt(mean_squared_error(training_predictions, Y_train))
    error=0
    train=np.array(Y_valid)
    for i in range (0, len(validation_predictions)):
        error = (validation_predictions[i]- train[i] )/train[i] + error
    error=(error*100)/len(validation_predictions)
    sl.write("Source Dataset: Walmart_Store_sales.csv")
    sl.write("Modèle: DNN")
    sl.write("L'erreur de validation: ",error)
    plt.plot(Y_valid)
    plt.plot(validation_predictions)
   
    #on visualise les valeurs réelles et prédites
    sl.set_option('deprecation.showPyplotGlobalUse', False)
    sl.pyplot()

