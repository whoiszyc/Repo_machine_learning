
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

#Given local path, find full path
def PathFinder(path):
    script_dir = os.path.dirname('__file__')
    fullpath = os.path.join(script_dir,path)
    return fullpath

#Read data_load_forecast for Deep Learning
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


# import tensorflow as tf  # @ZYC comment out
# tf.python.control_flow_ops = tf  # @ZYC comment out
from keras.layers import Input, Dense
from keras.layers import concatenate  # @ZYC changed from: from keras.layers import merge
from keras.models import Model
from keras import backend as K


class DenselyConnectedNetwork(object):
    '''

    '''

    def __init__(self, observ, hidden, output, num_layers, drop_out, boost):
        self.drop_out = drop_out
        self.boost = boost

        inputs = Input(shape=(observ,))
        x = Dense(hidden, activation='relu')(inputs)
        x = Dropout(drop_out)(x)
        #         interm_inputs = merge([inputs,x], mode='concat')
        #         if num_layers > 1:
        #             for i in range(num_layers-1):
        #                 x = Dense(hidden, activation='relu')(interm_inputs)
        #                 x = Dropout(drop_out)(x)
        #                 interm_inputs=merge([interm_inputs, x], mode='concat')
        #         predictions = Dense(output, activation='linear')(interm_inputs)
        if num_layers > 1:
            for i in range(num_layers - 1):
                x = Dense(hidden, activation='relu')(x)
                x = Dropout(drop_out)(x)
        interm_inputs = concatenate([x, inputs])  # @ZYC changed from: interm_inputs = merge([x, inputs], mode='concat')
        predictions = Dense(output, activation='linear')(interm_inputs)
        self.DeepNet = Model(input=inputs, output=predictions)
        self.DeepNet.compile(optimizer='rmsprop', loss=self.boosted_mean_squared_error)

    def boosted_mean_squared_error(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) * self.boost, axis=-1)

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
        plt.savefig('train_curve.png')


# from theano.tensor.nnet import relu  # @ZYC comment out
from torch.nn.functional import relu  # @ZYC add

def Interm_Distribution(layers,data):
    weights_bias=[]
    mean_matrix=[]
    std_matrix=[]
    for layer in layers:
        if type(layer) is Dense:
            weights_bias.append(layer.get_weights())
    m,n=np.array(weights_bias).shape
    previous_LayerOutput=data
    for i in range(m):
        LayerOutput = relu(np.dot(previous_LayerOutput,weights_bias[i][0])+np.array(weights_bias[i][1]))
        previous_LayerOutput=LayerOutput
        mean_matrix.append(np.mean(LayerOutput, axis=0))
        std_matrix.append(np.std(LayerOutput, axis=0))
    return mean_matrix,std_matrix

def write_Interm_Ditribution(means,stds, filepath,nodenames,hiddenstart='N'):
    fullpath=PathFinder(filepath)
    filehandler = open(fullpath,'w')
    for i in range(len(means)-1):
        for j in range(len(means[i])):
            row = []
            row.append(hiddenstart+str(i+1)+str(j))
            row.append(means[i][j])
            row.append(stds[i][j])
            filehandler.write(','.join(map(str, row))+'\n')
    filehandler.close()
    print('Interm Distribution Done!')


class NetTopology(object):
    def __init__(self, layers, normalized=False, mean=[], std=[], train_type='regular'):
        self.layers = layers
        self.input_dim = layers[0].get_config().get('batch_input_shape')[1]
        self.num_upper_layers = len(layers)
        self.mean = mean
        self.std = std
        self.train_type = train_type
        self.nodenames = []
        layernodename = []
        for i in range(0, self.input_dim):
            layernodename.append('N0' + str(i))
        layernodename.append('B0')
        self.nodenames.append(layernodename)

    def LayerWiseTransform(self, layer, layer_id, hiddenstart='N', writefile=False, filehandler=None, lastlayer=False):
        input_dim, output_dim = layer.get_weights()[0].shape
        if (lastlayer == True):
            activation = self.train_type
        else:
            activation = layer.get_config().get('activation')
        layernodename = []
        weights_bias = layer.get_weights()
        weights = weights_bias[0]
        bias = weights_bias[1]
        for i in range(0, output_dim):
            layernodename.append(hiddenstart + str(layer_id) + str(i))
        for i in range(0, output_dim):
            row = [layernodename[i], activation]
            if (layer_id == 1):
                adjustedweights = []
                for j in range(0, input_dim):
                    row.append(self.nodenames[-1][j])
                    if self.std[j] != 0:
                        adjustedweights.append(weights[j][i] / self.std[j])
                        row.append(weights[j][i] / self.std[j])
                    else:
                        adjustedweights.append(0)
                        row.append(0)
                row.append(self.nodenames[-1][-1])
                row.append(bias[i] - np.dot(np.array(adjustedweights), np.array(self.mean)))
            else:
                if input_dim == (len(self.nodenames[-1]) - 1):
                    for j in range(0, input_dim):
                        row.append(self.nodenames[-1][j])
                        row.append(weights[j][i])
                    row.append(self.nodenames[-1][-1])
                    row.append(bias[i])
                else:
                    for j in range(0, len(self.nodenames[-1]) - 1):
                        row.append(self.nodenames[-1][j])
                        row.append(weights[j][i])
                    adjustedweights = []
                    index_shift = len(self.nodenames[-1]) - 1
                    for j in range(0, len(self.nodenames[0]) - 1):
                        row.append(self.nodenames[0][j])
                        if self.std[j] != 0:
                            adjustedweights.append(weights[j + index_shift][i] / self.std[j])
                            row.append(weights[j + index_shift][i] / self.std[j])
                        else:
                            adjustedweights.append(0)
                            row.append(0)
                    row.append(self.nodenames[-1][-1])
                    row.append(bias[i] - np.dot(np.array(adjustedweights), np.array(self.mean)))

            if writefile:
                filehandler.write(','.join(map(str, row)) + '\n')
            else:
                print(','.join(map(str, row)))
        layernodename.append('B' + str(layer_id))
        self.nodenames.append(layernodename)

    def NetTransform(self, hiddenstart='N', writefile=False, filepath=None, overwrite=False):
        filehandler = None
        if writefile:
            fullpath = PathFinder(filepath)
            if overwrite is True:
                filehandler = open(fullpath, 'w')
            else:
                filehandler = open(fullpath, 'a')
        counter = 0
        for i in range(0, self.num_upper_layers):
            if type(self.layers[i]) is Dense:
                if (i == self.num_upper_layers - 1):
                    self.LayerWiseTransform(self.layers[i], counter + 1, hiddenstart, writefile, filehandler, True)
                else:
                    self.LayerWiseTransform(self.layers[i], counter + 1, hiddenstart, writefile, filehandler, False)
                counter = counter + 1
        print('Network Dumping Done!')


def writeNormInfo(PD_Data,mean_DNN,std_DNN,problem):
    fullpath = PathFinder('Network_Normalization_'+problem+'.txt')
    filehandler = open(fullpath,'w')
    Headers=list(PD_Data.columns.values)
    for i in range(len(Headers)):
        filehandler.write('N0'+str(i)+','+Headers[i].replace(': ', ',')+','+str(mean_DNN[i])+','+str(std_DNN[i])+'\n')
    filehandler.close()
    print('Normalization File Complete!')


def inPutFiles(PD_Data,PD_Label,Input_Num,Output_Num,Hidden_Head, Net_Depth, Problem,Type):#Type in {Regular,Delta}
    Headers=list(PD_Data.columns.values)+list(PD_Label.columns.values)
    fullpath=PathFinder("Headers_RDDL_"+Problem+".txt")
    filehandler = open(fullpath,'w')
    for i in range(Input_Num):
        filehandler.write('N0'+str(i)+','+Headers[i].replace(': ', ',')+'\n')
    for i in range(Output_Num):
        filehandler.write(Hidden_Head+str(Net_Depth)+str(i)+','+Type+','+Headers[i+Input_Num].replace(':', ',')+'\n')
    filehandler.close()
    print('Headers File Complete!')


from matplotlib.ticker import ScalarFormatter
def performanceViz(Data,True_Label,Pred_Label,horizon):
    _,dim=Data.shape
    fig, axes = plt.subplots(nrows=dim, ncols=1, figsize=(8, 6))
    y_formatter = ScalarFormatter(useOffset=False)
    for i in range(dim):
        for j in range(horizon):
            axes[i].plot([j,j+1],[Data[j,i],True_Label[j,i]],'r-',lw=1.5)
            axes[i].plot([j,j+1],[Data[j,i],Pred_Label[j,i]],'b-.',lw=1.5)
            axes[i].yaxis.set_major_formatter(y_formatter)
            axes[i].locator_params(axis='y',nbins=5)
    fig.subplots_adjust(hspace=0.4)
    plt.savefig('Comparison.png')


def performanceCheck(DNN,Test_Data,Test_Label,mean_DNN,std_DNN):
    Pred_Label = DNN.Test(Test_Data,True,mean_DNN,std_DNN)
    print("Complete testing")
    Feed_Data=Test_Data[:20,6:]
#     performanceViz(Feed_Data[:,[0,1,2,5,6,7,8]],Test_Label[:20,[0,1,2,5,6,7,8]],Pred_Label[:20,[0,1,2,5,6,7,8]],20)
    performanceViz(Feed_Data[:],Test_Label[:20],Pred_Label[:20],20)



def trainNetForData(Datapath,Labelpath,node_size,layer_num,train_type,problem):
    # read data
    # feature data including actions and states         [u_t, s_t; u_t+1, s_t+1]
    # target data only including states at next step    [   s_t+1;     s_t+2]
    PD_Data = ReadData(Datapath)
    PD_Label = ReadData(Labelpath)

    # transfer data into matrices (tensor)
    Full_Data=PD_Data.as_matrix()
    Full_Label=PD_Label.as_matrix()

    # Randomly permute a sequence, or return a permuted range.
    indecs=np.random.permutation(len(Full_Data))
    Full_Data=Full_Data[indecs]
    Full_Label=Full_Label[indecs]

    # get dimension
    m_data,n_data=Full_Data.shape
    m_label,n_label=Full_Label.shape

    inPutFiles(PD_Data,PD_Label,n_data,n_label,'D',layer_num+1,problem,train_type)#Could be Delta

    # split data into training set and testing set
    Train_Data = Full_Data[:int(m_data*0.9)]
    Train_Label = Full_Label[:int(m_data*0.9)]
    Label_Weights = (1.0/np.square(np.max(Train_Label,axis=0)+1))*10
    Test_Data = Full_Data[int(m_data*0.9):]
    Test_Label = Full_Label[int(m_data*0.9):]

    DNN=DenselyConnectedNetwork(n_data,node_size,n_label,layer_num,0.1,Label_Weights)
    mean_DNN,std_DNN=DNN.Train(Train_Data, Train_Label,10,True)
    DNN.ShowHistory()

    writeNormInfo(PD_Data,mean_DNN,std_DNN,problem)
    Topo=NetTopology(DNN.GetModel().layers,True, mean_DNN, std_DNN,train_type)
    Topo.NetTransform('D',True, "Network_RDDL_"+problem+".txt",True)
#     mean_Matrix,std_Matrix=Interm_Distribution(DNN.GetModel().layers,Train_Data)
#     write_Interm_Ditribution(mean_Matrix,std_Matrix,'Interm_Distribution_RDDL_'+problem+'.txt','D')
    return DNN,Test_Data,Test_Label,mean_DNN,std_DNN,Topo



Datapath="HVAC_COMPLEX_DATA.txt"
Labelpath="HVAC_COMPLEX_LABEL.txt"
DNN,Test_Data,Test_Label,mean_DNN,std_DNN,Topo=trainNetForData(Datapath,Labelpath,32,1,'regular','HVAC_COMPLEX')
performanceCheck(DNN,Test_Data,Test_Label,mean_DNN,std_DNN)
DNN.DeepNet.save('HVAC.h5')


import onnxmltools
import onnx
onnx_model = onnxmltools.convert_keras(DNN.DeepNet)
onnx.save(onnx_model, 'model_HVAC.onnx')

