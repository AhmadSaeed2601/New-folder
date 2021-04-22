  
# streamlit run your_script.py --server.maxUploadSize=1028
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import scipy as sci


import tensorflow as tf
from tensorflow.keras.layers import Dense

from scipy import io
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import autoplotter as app_run


# Page Layout
st.set_page_config(page_title='POD Coefficient Prediction Application', layout='wide')


st.write("""
# Multilayer Perceptron to Predict the PCA Coefficients
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe

st.sidebar.header('Upload your data file')
uploaded_file = st.sidebar.file_uploader('Upload your mat file', type = ['mat'])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")


# Sidebar ---- Parameter setting 

st.sidebar.header('Parameter Setting')
number_of_neurons = st.sidebar.slider('Number of Neurons', 1,64,15,1)

time_step = st.sidebar.slider('Time Step Between Snapsots', 0.01, 0.1, 0.05, 10.)
total_snaps = st.sidebar.slider('Total steps', 1,1000,132,50)
for_test_snaps = st.sidebar.slider('Testings time steps', 10,500,30,20)
for_train_snaps = total_snaps- for_test_snaps

st.sidebar.subheader('Select Optimizer')
optimizer = st.sidebar.selectbox('Select the Optimizer for Training', ('Adam', 'SGD', 'Adamax', 'Nadam'))
learning_rate = st.sidebar.select_slider('Set Learning Rate', value=0.01,options=[0.01,0.001])
no_of_epoch = st.sidebar.slider('Epoch', 10,10000,100)

st.sidebar.subheader('POD Coefficients')
coeff_no = st.sidebar.slider('Number of Coefficient', 1,15,8,1)
plot_coeff = st.sidebar.slider('Plot no. of Coefficient', 0,15,1,1)



# Displays the dataset
st.subheader('Dataset')

def get_optimizer(optimizer):
    optimiser = None
    if optimizer == 'Adam':
        optimiser = 'adam'

    elif optimizer == 'SGD':
        optimiser = 'SGD'

    elif optimizer == 'Nadam':
        optimiser = 'nadam'

    else :
        optimiser = 'adamax'
    return optimiser

def build_model(data):

    st.markdown('A model is being built to predict the **Future Time Steps (Y_train, Y_test)**:')
    
    model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(number_of_neurons,activation='tanh'),
     tf.keras.layers.Dense(number_of_neurons, activation='tanh'),
     tf.keras.layers.Dense(number_of_neurons, activation='tanh'),
     tf.keras.layers.Dense(coeff_no)
    ])

    
    optimiser = get_optimizer(optimizer)

    model.compile(optimizer=optimiser,
              loss='mse')

    #t_board_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
    history = model.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=no_of_epoch)
    model.evaluate(X_test, Y_test)
    #st.write(model.summary())

    st.subheader('Training and Validation Loss')
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    st.write(hist)



    fig = go.Figure()

    fig.add_trace(go.Scatter(x = hist['epoch'] , y = history.history['loss'],
                                                 mode= 'lines', name= 'Train_Loss'))

    fig.add_trace(go.Scatter(x = hist['epoch'] , y = history.history['val_loss'],
                                                 mode= 'lines', name= 'val_Loss'))

    fig.update_layout(xaxis_title = 'Epoch', yaxis_title = 'Loss')   

    st.plotly_chart(fig)

    #fig, ax = plt.subplots()
    #ax = plt.plot(history.history['loss'], alpha=0.8, label= 'Train_Loss')
    #ax = plt.plot(history.history['val_loss'], alpha=0.8, label= 'val_Loss')
    #ax = plt.legend()
    #ax = plt.xlabel('Epoch')
    #ax = plt.ylabel(f'Loss')
    #st.pyplot(fig=fig)


    return model



if uploaded_file is not None:
    @st.cache
    def load_data(uploaded_file):
        data = io.loadmat(uploaded_file)
        return data

    data = load_data(uploaded_file)
    PALL = data['PALL']
    X = PALL
    nx, ny = 194, 256
    Xavg = np.mean(X,axis=1)                     # Compute mean
    data = X - np.tile(Xavg,(X.shape[1],1)).T    # subtract mean
   
    _,_,V = np.linalg.svd(data, full_matrices=0)     # Compute Coefficients



    st.subheader('Mean')
    t = np.linspace(1, time_step*(for_train_snaps), for_train_snaps)
    avg = np.reshape(Xavg, (nx, ny))
    st.write(avg)
    st.write(t)
    st.write(for_train_snaps)
    # PLot surface Mean
    st.subheader('Surface Mean Mode')
    fig = px.line(y = avg[0,:])
    st.plotly_chart(fig)


    # V = preprocessing.normalize(V)
    
    V = V.transpose()
    V = V[:,0:coeff_no]

    st.subheader('POD Coefficients')
    st.write(V)

    scaler = MinMaxScaler()
    Xtrain = V[:-1,:]
    Ytrain = V[1:,:]
    

    #Xtrain = scaler.fit_transform(Xtrain)
    #Ytrain= scaler.fit_transform(Ytrain)

    

    # Test and Train Split 
    Total_time_steps = total_snaps
    for_train = Total_time_steps - for_test_snaps

    X_train = Xtrain[:for_train,:]
    Y_train = Xtrain[1:for_train+1,:]
    X_test = Ytrain[for_train:Total_time_steps-1,:]
    Y_test = Ytrain[for_train+1:Total_time_steps,:]

    #X_train = Xtrain[:101,:]
    #Y_train = Xtrain[1:102,:]
    #X_test = Ytrain [101:131,:]
    #Y_test = Ytrain [102:132,:]

    data = X_train, X_test, Y_train, Y_test

    st.info('Press to Start Training Neural Network.')
    if st.button('Press to Start'):
        model = build_model(data)

    #if st.button('Plot Results'):
        pred_train, pred_test =  model.predict(X_train), model.predict(X_test)
        #pred_train, pred_test = scaler.inverse_transform(pred_train), scaler.inverse_transform(pred_test)

        pred_data = np.concatenate((pred_train, pred_test), axis=0)


        # Plotting -------------------

        #fig, ax = plt.subplots()
        #t = np.linspace(1,0.05*100, len(pred_train[:,plot_coeff]))
        #ax =  plt.scatter(t, pred_train[:,plot_coeff], alpha=0.8, label= 'Predicted')
        #ax = plt.plot(t, Y_train[:,plot_coeff], alpha=0.8, label= 'True')
        #ax = plt.legend()
        #ax = plt.xlabel('Time')
        #ax = plt.ylabel(f'POD coefficient {coeff_no}')


        # Plotly Plotting
        cols = []
        for i in range(coeff_no):
            cols.append("Coefficient_" + str(i+1))

        df_pred_train = pd.DataFrame(pred_train, columns= cols)
        df_pred_test = pd.DataFrame(pred_test, columns= cols)
        df_Y_train = pd.DataFrame(Y_train, columns= cols)
        df_Y_test = pd.DataFrame(Y_test, columns= cols)

        t_train = np.linspace(0,time_step*(for_train_snaps), for_train_snaps)
        t_test =  np.linspace(t_train[-1], t_train[-1] + time_step*(for_test_snaps), for_test_snaps)


        
        fig = make_subplots(rows= 2, cols=2, subplot_titles=['Training data'])
        fig.add_trace(go.Scatter(x= t_train, y= df_pred_train.Coefficient_1, mode = 'markers', line=dict(color='red', width=2)), row = 1, col =1)
        fig.add_trace(go.Scatter(x= t_train, y= df_Y_train.Coefficient_1, mode = 'lines', line=dict(color='blue', width=2)), row = 1, col =1)

        fig.add_trace(go.Scatter(x= t_train, y= df_pred_train.Coefficient_3, mode = 'markers', line=dict(color='red', width=2)), row = 1, col =2)
        fig.add_trace(go.Scatter(x= t_train, y= df_Y_train.Coefficient_3, mode = 'lines', line=dict(color='blue', width=2)), row = 1, col =2)

        fig.add_trace(go.Scatter(x= t_train, y= df_pred_train.Coefficient_5, mode = 'markers', line=dict(color='red', width=2)), row = 2, col =1)
        fig.add_trace(go.Scatter(x= t_train, y= df_Y_train.Coefficient_5, mode = 'lines', line=dict(color='blue', width=2)), row = 2, col =1)

        fig.add_trace(go.Scatter(x= t_train, y= df_pred_train.Coefficient_7, mode = 'markers', line=dict(color='red', width=2)), row = 2, col =2)
        fig.add_trace(go.Scatter(x= t_train, y= df_Y_train.Coefficient_7, mode = 'lines', line=dict(color='blue', width=2)), row = 2, col =2)

        st.plotly_chart(fig)
        #st.pyplot(fig=fig)




         # Testing data
        fig = make_subplots(rows= 2, cols=2, subplot_titles=['Validation Coefficients'])
        fig.add_trace(go.Scatter(x= t_test, y= df_pred_test.Coefficient_1, mode = 'markers', line=dict(color='red', width=2)), row = 1, col =1)
        fig.add_trace(go.Scatter(x= t_test, y= df_Y_test.Coefficient_1, mode = 'lines', line=dict(color='blue', width=2)), row = 1, col =1)

        fig.add_trace(go.Scatter(x= t_test, y= df_pred_test.Coefficient_3, mode = 'markers', line=dict(color='red', width=2)), row = 1, col =2)
        fig.add_trace(go.Scatter(x= t_test, y= df_Y_test.Coefficient_3, mode = 'lines', line=dict(color='blue', width=2)), row = 1, col =2)

        fig.add_trace(go.Scatter(x= t_test, y= df_pred_test.Coefficient_5, mode = 'markers', line=dict(color='red', width=2)), row = 2, col =1)
        fig.add_trace(go.Scatter(x= t_test, y= df_Y_test.Coefficient_5, mode = 'lines', line=dict(color='blue', width=2)), row = 2, col =1)

        fig.add_trace(go.Scatter(x= t_test, y= df_pred_test.Coefficient_7, mode = 'markers', line=dict(color='red', width=2)), row = 2, col =2)
        fig.add_trace(go.Scatter(x= t_test, y= df_Y_test.Coefficient_7, mode = 'lines', line=dict(color='blue', width=2)), row = 2, col =2)

        st.plotly_chart(fig)

        
