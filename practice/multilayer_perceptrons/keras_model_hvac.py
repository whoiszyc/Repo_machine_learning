
import random
import os
import json
import sys
import string
import unicodedata
from tqdm import tqdm
import pandas as pd

#Numpy and Scipy
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.spatial import distance

#KERAS
from keras.layers import Dense, Activation,Dropout
from keras.models import Sequential
from keras.regularizers import l1,l2
from keras.models import load_model

# import tensorflow as tf  # @ZYC comment out
# tf.python.control_flow_ops = tf  # @ZYC comment out
from keras.layers import Input, Dense
from keras.layers import concatenate  # @ZYC changed from: from keras.layers import merge
from keras.models import Model
from keras import backend as K

# from theano.tensor.nnet import relu  # @ZYC comment out
from torch.nn.functional import relu  # @ZYC add

#Given local path, find full path
def PathFinder(path):
    script_dir = os.path.dirname('__file__')
    fullpath = os.path.join(script_dir,path)
    return fullpath

#Read Data for Deep Learning
def ReadData(path):
    fullpath=PathFinder(path)
    return pd.read_csv(fullpath, sep=',', header=0)

#Input Normalization
def Normalize(features, mean = [], std = []):
    if mean == []:
        mean = np.mean(features, axis = 0)
        std = np.std(features, axis = 0)
#     print std
#     print std[:,None]
    new_feature = (features.T - mean[:,None]).T
    new_feature = (new_feature.T / std[:,None]).T
    new_feature[np.isnan(new_feature)]=0
#     print new_feature
    return new_feature, mean, std





class DenselyConnectedNetwork(object):
    '''

    '''

    def __init__(self, observ, hidden, output, num_layers, drop_out, boost):
        self.drop_out = drop_out
        self.boost = boost

        inputs = Input(shape=(observ,))
        x = Dense(hidden, activation='relu')(inputs)
        # x = Dropout(drop_out)(x)

        if num_layers > 1:
            for i in range(num_layers - 1):
                x = Dense(hidden, activation='relu')(x)
                # x = Dropout(drop_out)(x)
        ## OP 1:
        # interm_inputs = concatenate([x, inputs])  # @ZYC changed from: interm_inputs = merge([x, inputs], mode='concat')
        # predictions = Dense(output, activation='linear')(interm_inputs) # @ZYC comment out
        ## OP 2:
        predictions = Dense(output, activation='linear')(x)  # @ZYC add
        self.DeepNet = Model(input=inputs, output=predictions)

        ## OP 1:
        # self.DeepNet.compile(optimizer='rmsprop', loss=self.boosted_mean_squared_error)
        ## OP 2:
        self.DeepNet.compile(optimizer='rmsprop', loss='mean_squared_error')


    def Train(self, data, label, epoch, normalize=False):
        mean = []
        std = []
        if normalize:
            normalized_data, mean, std = Normalize(data)
        else:
            normalized_data = data
        self.history = self.DeepNet.fit(normalized_data, label, validation_split=0.1, batch_size=128, nb_epoch=epoch)
        return mean, std


    def Test(self, datapoint, normalize=False, mean=[], std=[]):
        if normalize:
            normalized_datapoint, _, _ = Normalize(datapoint, mean, std)
        else:
            normalized_datapoint = datapoint
        return self.DeepNet.predict(normalized_datapoint, batch_size=128, verbose=0)


    def LoadModel(self, modelpath):
        self.DeepNet = load_model(modelpath)


    def Save(self, modelpath):
        self.DeepNet.save(modelpath)


    def GetModel(self):
        return self.DeepNet


    def ShowHistory(self):
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig('train_curve.png')





def trainNetForData(Datapath,Labelpath,node_size,layer_num,train_type,problem,normalize=False):
    # read data
    # feature data including actions and states         [u_t, s_t; u_t+1, s_t+1]
    # target data only including states at next step    [   s_t+1;     s_t+2]
    PD_Data = ReadData(Datapath)
    PD_Label = ReadData(Labelpath)

    # transfer data into matrices (tensor)
    Full_Data=PD_Data.as_matrix()
    Full_Label=PD_Label.as_matrix()

    # # Randomly permute a sequence, or return a permuted range.
    # indecs=np.random.permutation(len(Full_Data))
    # Full_Data=Full_Data[indecs]
    # Full_Label=Full_Label[indecs]

    # get dimension
    m_data,n_data=Full_Data.shape
    m_label,n_label=Full_Label.shape

    # inPutFiles(PD_Data,PD_Label,n_data,n_label,'D',layer_num+1,problem,train_type)#Could be Delta

    # split data into training set and testing set
    Train_Data = Full_Data[:int(m_data*0.9)]
    Train_Label = Full_Label[:int(m_data*0.9)]
    Label_Weights = (1.0/np.square(np.max(Train_Label,axis=0)+1))*10
    Test_Data = Full_Data[int(m_data*0.9):]
    Test_Label = Full_Label[int(m_data*0.9):]

    # train the neural networks
    DNN=DenselyConnectedNetwork(n_data,node_size,n_label,layer_num,0.1,Label_Weights)
    mean_DNN,std_DNN=DNN.Train(Train_Data, Train_Label,10,normalize)
    DNN.ShowHistory()

    return DNN,Test_Data,Test_Label,mean_DNN,std_DNN






if __name__=="__main__":
    # Give path to the data
    Datapath="data_hvac/HVAC_COMPLEX_DATA.txt"
    Labelpath="data_hvac/HVAC_COMPLEX_LABEL.txt"

    # Build Neural networks and perform training
    DNN,Test_Data,Test_Label,mean_DNN,std_DNN=trainNetForData(Datapath,Labelpath,32,1,'regular','HVAC_COMPLEX',False)

    # Test
    Pred_Label = DNN.Test(Test_Data,False,mean_DNN,std_DNN)
    # print("Complete testing")
    # Feed_Data = Test_Data[:20,6:]
    # performanceViz(Feed_Data[:],Test_Label[:20],Pred_Label[:20],20)

    # t, _ = Pred_Label.shape
    t = (20, 200)
    var = 5  # from 0 to 12 are temperature from room 1 to 6 and air volume from room 1 to 6
    plt.figure()
    plt.plot(range(t[0], t[1]), Test_Label[t[0]:t[1], var], 'r')
    plt.plot(range(t[0], t[1]), Pred_Label[t[0]:t[1], var], 'b-.')
    plt.title('Prediction using Neural Networks')
    # plt.savefig('Comparison_Keras.png')

    # Save model
    # DNN.DeepNet.save('HVAC.h5')

