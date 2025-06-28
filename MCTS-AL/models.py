import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,ModelCheckpoint
from tensorflow.keras.callbacks import Callback
import seaborn as sns
from sklearn import metrics
from scipy.stats import pearsonr
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow.compat.v1 as tf
from Voxelization import Voxelization
tf.disable_v2_behavior()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

"""
CNN models used to predict the properties of designs

:param x: inputs of the dataset
       y: labels of the dataset
       rdnum: which iteration of AL is
       lr: learning rate in the train
       num: numbers of the models
"""
class models:
    def __init__(self, x, y, rdnum, lr = 0.001, num = 5):
        self.input_x = x
        self.input_y = y
        self.lr = lr
        self.rdnum = rdnum
        self.round_name = f'Round{rdnum}'
        self.create_folder()
        self.num = num
    
    def create_folder(self):
        current_dir = os.getcwd()
        self.model_folder = current_dir + f'/models/{self.round_name}'
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
    
    """
    Train process of a single CNN model
    
    :param index_random: indexes of data used to train this model
           i: number of the model
    :return hist: history of the train process
            model:
            X_test: inputs of the test dataset
            Y_test: labels of the test dataset
            R2: pearson r^2
            MAE: mean absolute error
    """
    def single_train(self, index_random, i):
        X = Voxelization(self.input_x)
        X = np.expand_dims(X, axis=-1)
        Y = self.input_y
        
        ind = index_random[round(i*len(index_random)/5):round((1+i)*len(index_random)/5)]
        ind2 = np.setdiff1d(index_random, ind)
        X_train, X_test, Y_train, Y_test = X[ind2],X[ind], Y[ind2],Y[ind]
        inputs = keras.Input((60, 60, 60, 1))
        
        #convolutional layers
        x = layers.Conv3D(filters=16, kernel_size=2, activation="elu",padding='same')(inputs)
        x = layers.MaxPool3D(pool_size=2, padding='same')(x)
        x = layers.Conv3D(filters=8, kernel_size=2, activation="elu",padding='same')(x)
        x = layers.MaxPool3D(pool_size=2, padding='same')(x)
        x = layers.Conv3D(filters=4, kernel_size=2, activation="elu",padding='same')(x)
        x = layers.MaxPool3D(pool_size=2, padding='same')(x)
        
        #fully connected layers
        x = layers.Flatten()(x)
        x = layers.Dense(units=128, activation="elu")(x)
        x = layers.Dense(units=64, activation="elu")(x)
        x = layers.Dense(units=32, activation="elu")(x)
        
        outputs = layers.Dense(units=1, activation="linear")(x)
        model = keras.Model(inputs, outputs, name="3dcnn")
        
        mc = ModelCheckpoint(f'{self.model_folder}/{i}_temp.h5', monitor='val_loss',
                             mode='min', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mean_absolute_error')
        hist = model.fit(X_train, Y_train, batch_size=32, epochs=5000, 
                         validation_data=(X_test, Y_test), callbacks=[early_stop,mc])
        
        model = keras.models.load_model(f'{self.model_folder}/{i}_temp.h5')
        R2,MAE = self.mar_r2(model, X_test, Y_test)
    
        return [hist, model, X_test, Y_test, R2, MAE]
    
    """
    Train an ensemble of models
    """
    def ensembled_training(self):
        pd.DataFrame(np.empty(0)).to_csv(f'{self.model_folder}/model_performance.csv')
        
        index_random=np.arange(len(self.input_y))
        np.random.shuffle(index_random)
        
        hists = []
        for i in range(self.num):
            temp_result = self.single_train(index_random, i)
            
            temp_R2 = temp_result[-2]
            repeats = 0
            #repeat the training for 5 times, and record the best one
            while temp_R2 < 0.995 and repeats < 1:
                repeats += 1
                result = self.single_train(index_random, i)
                R2 = result[-2]
                if R2 > temp_R2:
                    temp_result = result
            
            temp_model = temp_result[1]
            temp_hist = temp_result[0]
            temp_model.save(f'{self.model_folder}/{i}.h5')
            self.plot_performance(result)
            hists.append(temp_hist)
        self.hists = hists
    
    """
    Plot and save the performance of the models
    
    :param result: a list of hist, model, X_test, Y_test, R2 and MAE
    """
    def plot_performance(self, result):
        [hist, model, X_test, Y_test, R2, MAE] = result
        perform_list = pd.read_csv(f'{self.model_folder}/model_performance.csv')
        Y_pred = model.predict(X_test.reshape(len(X_test),60,60,60,1))
        ###plot R2 and MAE of test data
        plt.figure()
        sns.set()
        sns.regplot(x=Y_pred, y=Y_test, color='k')
        plt.title(('R2:',R2,'MAE:',MAE))
        y_test = pd.DataFrame(Y_test)
        y_test.columns= ['ground truth']
        y_pred = pd.DataFrame(Y_pred)
        y_pred.columns= ['prediction']
        R2MAE = pd.DataFrame([R2,MAE])
        R2MAE.columns= ['R2&MAE']
        perform_list = pd.concat((perform_list,y_test,y_pred,R2MAE),axis=1)
        perform_list.drop([perform_list.columns[0]],axis=1, inplace=True)
        perform_list.to_csv(f'{self.model_folder}/model_performance.csv')
    
    """
    Calculate the R2 and MAE of the model
    
    """
    def mar_r2(self,model,X_test,y_test):
        y_pred = model.predict(X_test.reshape(len(X_test),60,60,60,1))
        R2 = pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
        R2 = np.asarray(R2).round(5)
        MAE = metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
        return R2,MAE
    
    """
    If there are already trained models, load them without retrain an ensemble
    of new onse
    
    """
    def load_models(self):
        models = []
        for i in range(self.num):
            model = keras.models.load_model(f'{self.model_folder}/{i}.h5')
            models.append(model)
        self.models = models
    
    """
    Predict the property of input S
    
    :param S: structure or designs
    :return pred_all: average of the predictions of the models
    """
    def ensembled_prediction(self, S):##E
        pred_all = 0
        for model in self.models:
            temp = model.predict(S.reshape(len(S),60,60,60,1))
            pred_all += temp
        pred_all /= self.num
        return pred_all 

