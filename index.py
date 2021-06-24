import streamlit as sl
import pandas as pd
import matplotlib.pyplot as plt

#la fonction qui permet de visualiser les résultats du modèle sous forme de plots
def visualise_result():
   pass

# Le titre de l'app
sl.title("Deep Learning for Time Series Forecasting App")
#une barre séparente horizentale
sl.markdown("***")

#Description de l'app
sl.header("Notre app permet d'utiliser un modèle de Deep Learning pour prédire des valeurs réelles dans une série temoporelle")
#Sauter d'espace
sl.markdown("#")
if(sl.button("Charger le Data Set")):
    weekly_sales = pd.read_csv('Walmart_Store_sales.csv')
    sl.write(weekly_sales)
    sl.write("Les dimensions du Data Set: ",weekly_sales.shape)


#Créer un menu latéral pour le choix du modèle et son paramètrage
select_model = sl.sidebar.selectbox("Choisir le modèle: ", ('RNN', 'LSTM', 'DNN'))
#paramètrage:
if(select_model=='LSTM'):
    select_learning_rate = sl.sidebar.text_input("Le taux d'apprentissage: ", 'Recommandé(0.001)')
    select_epochs = sl.sidebar.text_input("Le nombre d'epochs(ou steps pour le DNN): ", 'Recommandé(200)')
    select_batch_size = sl.sidebar.text_input("La taille de batch: ", 'Recommandé(32)')
elif(select_model=='RNN'):
    select_learning_rate = sl.sidebar.text_input("Le taux d'apprentissage: ", 'Recommandé(0.001)')
    select_epochs = sl.sidebar.text_input("Le nombre d'epochs(ou steps pour le DNN): ", 'Recommandé(300)')
    select_batch_size = sl.sidebar.text_input("La taille de batch: ", 'Recommandé(200)')
else:
    select_learning_rate = sl.sidebar.text_input("Le taux d'apprentissage: ", 'Recommandé(0.001)')
    select_epochs = sl.sidebar.text_input("Le nombre d'epochs(ou steps pour le DNN): ", 'Recommandé(100)')
    select_batch_size = sl.sidebar.text_input("La taille de batch: ", 'Recommandé(10)')





sl.sidebar.markdown('#')
#Boutton pour visualiser les résultats du modèle fitting et validation
if(sl.sidebar.button("Visualiser les résultats")):
    if (select_model=='LSTM'):
        from lstm import * #on importe les fonctions du fichier LSTM pour l'implémentation
        sl.info("Veuillez patienter, cette procédure prend des minutes!")
        implement_model(select_model, select_learning_rate, select_epochs, select_batch_size)
        
    elif (select_model=='RNN'):
         from rnn import *
         sl.info("Veuillez patienter, cette procédure prend des minutes!")
         implement_model_rnn(select_learning_rate, select_epochs, select_batch_size)
         
    elif (select_model=='DNN'):
        from dnn import *
        sl.info("Veuillez patienter, cette procédure prend du minutes!")
        implement_model_dnn(select_model, select_learning_rate, select_epochs, select_batch_size)
       
        
