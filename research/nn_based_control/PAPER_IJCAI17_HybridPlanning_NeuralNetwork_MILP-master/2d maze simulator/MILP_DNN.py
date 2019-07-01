import random
import os
import json
import sys
import string
import unicodedata

#Numpy and Scipy
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.spatial import distance

#KERAS
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.regularizers import l1,l2
from keras.models import load_model

'''
Created on Oct 6, 2016

This class has two construction functions.
1. From data
2. From initial settings

Example of 2D maze setting

initial_setting = {
    "maze_bound"        : (0,0,10,10),             #Continous state bound
    "obstacles"         : [(1,1,3,3),(5,4,6,6.5)], #Some obstacles that never crosspassing
    "current_state"     : (0,0,0,1),               #Current State in X,Y and If in Jail
    "action_range"      : (-0.5,0.5),              #The effective action range
    "action_dim"        : 2,                       #Dimension of Actions
    "goal_state"        : (10,10,0,1),             #The goal state to finish running
    "jail_location"     : (-1,-1,1,0),             #Jain location
    "deadend_toleration": 2                        #How many step left after getting into jail
                   }
@author: wuga
'''

class DeterministicMaze(object):

    def __init__(self, setting):
        self.__dict__.update(setting)
        self.backup = setting
        self.X_AXIS = 0

    def Reset(self):
        backup=self.backup
        self.__dict__.update(backup)
        self.backup = backup
        self.X_AXIS = 0

    def UpdateState(self,new_state):
        self.current_state = new_state

    def Apply(self, action):
        if not self.current_state[2]:
            x_axis=self.current_state[0]
            y_axis=self.current_state[1]
            x_axis=self.current_state[0]+action[0]
            if x_axis>self.maze_bound[2]:
                x_axis=self.maze_bound[2]
            if x_axis<self.maze_bound[0]:
                x_axis=self.maze_bound[0]
            y_axis=self.current_state[1]+action[1]
            if y_axis>self.maze_bound[3]:
                y_axis=self.maze_bound[3]
            if y_axis<self.maze_bound[1]:
                y_axis=self.maze_bound[1]
            self.UpdateState((x_axis,y_axis,0,1))
            if self.Collision():
                self.UpdateState(self.jail_location)
        else:
            self.deadend_toleration=self.deadend_toleration-1
        return self.current_state

    def StateIn(self, area, position):
        if (position[0] >= area[0]) and (position[0] <= area[2]) and \
        (position[1] >= area[1]) and (position[1] <= area[3]):
            return True
        else:
            return False

    def Collision(self):
        for area in self.obstacles:
            if self.StateIn(area, self.current_state):
                return True
        return False

    def IfGameEnd(self):
        if (self.current_state[0] == self.goal_state[0]) and \
        (self.current_state[1] == self.goal_state[1]) and \
        (self.current_state[2] == self.goal_state[2]):
            #print 'Get goal state! system resetted'
            self.Reset()
        if (self.deadend_toleration == 0):
            self.Reset()

    def DeltaDistance(self,new_state,old_state):
        delta=[]
        delta.append(new_state[0]-old_state[0])
        delta.append(new_state[1]-old_state[1])
        if (new_state[2]==1):
            delta.append(1)
            delta.append(0)
        else:
            delta.append(0)
            delta.append(1)
        return delta

    def GetCurrentState(self):
        return self.current_state



initial_setting = {
    "maze_bound"        : (0,0,10,10),             #Continous state bound
    "obstacles"         : [(1,1,3,3),(5,4,6,6.5)], #Some obstacles that never crosspassing
    "current_state"     : (0,0,0,1),               #Current State in X,Y and If in Jail
    "action_range"      : (-0.5,0.5),              #The effective action range
    "goal_state"        : (10,10,0,1),             #The goal state to finish running
    "jail_location"     : (-1,-1,1,0),             #Jain location
    "deadend_toleration": 2,                       #How many step left after getting into jail
    "action_dim"        : 2                        #Dimension of Actions
    }

# Maze=DeterministicMaze(initial_setting)
# print Maze.current_state
# action1=(0,11)
# Maze.Apply(action1)
# action2=(1,10)
# Maze.Apply(action2)
# print Maze.current_state


#Given local path, find full path
def PathFinder(path):
    script_dir = os.path.dirname('__file__')
    fullpath = os.path.join(script_dir,path)
    return fullpath

#Read data_load_forecast for Deep Learning
def ReadData(path):
    fullpath=PathFinder(path)
    return genfromtxt(fullpath, delimiter=',')


class RandomWalk(object):
    def __init__(self, actiondim, ranges):
        self.ranges = ranges
        self.actiondim = actiondim

    def Go(self):
        stride = []
        for i in range(self.actiondim):
            stride.append(random.triangular(self.ranges[0], self.ranges[1], 0.2))
        print(stride)
        return stride

Maze=DeterministicMaze(initial_setting)
Planner=RandomWalk(Maze.action_dim,Maze.action_range)
for i in range(10):
    Planner.Go()


class DataGenerator(object):
    def __init__(self, model):
        self.maze = model
        self.planner = RandomWalk(self.maze.action_dim, self.maze.action_range)

    def WriteData(self, size, datapath, labelpath):
        fulldatapath = PathFinder(datapath)
        fulllabelpath = PathFinder(labelpath)
        datafile = open(fulldatapath, 'w')
        labelfile = open(fulllabelpath, 'w')
        for i in range(0, size):
            self.maze.IfGameEnd()
            action = self.planner.Go()
            old_state = Maze.GetCurrentState()
            self.maze.Apply(action)
            new_state = self.maze.GetCurrentState()
            Data = list(old_state) + action
            # Label=list(new_state)
            Label = self.maze.DeltaDistance(new_state, old_state)
            datafile.write(','.join(map(str, Data)) + '\n')
            labelfile.write(','.join(map(str, Label)) + '\n')
        datafile.close()
        labelfile.close()

Maze=DeterministicMaze(initial_setting)
Generator=DataGenerator(Maze)
Datapath="StateActionData.txt"
Labelpath="StateActionLabel.txt"
Generator.WriteData(100000,Datapath,Labelpath)


class FullyConnectedDeepNet(object):
    '''

    '''

    def __init__(self, observ, hidden, output, num_layers, drop_out):
        self.drop_out = drop_out
        self.DeepNet = Sequential()
        # self.Autoencoder.add(Dropout(0.5, input_shape=(observ,)))
        self.DeepNet.add(Sequential([Dense(hidden, input_dim=observ), Activation('relu')]))
        self.DeepNet.add(Dropout(drop_out))
        if num_layers > 1:
            for i in range(num_layers - 1):
                self.DeepNet.add(Sequential([Dense(hidden, input_dim=hidden), Activation('relu')]))
                self.DeepNet.add(Dropout(drop_out))
        self.DeepNet.add(Dense(output, input_dim=hidden))
        self.DeepNet.compile(optimizer='rmsprop', loss='mse')

    def Train(self, data, label, epoch):
        self.DeepNet.fit(data, label, nb_epoch=epoch)

    def Test(self, datapoint):
        return self.DeepNet.predict(datapoint, batch_size=128, verbose=0)

    def LoadModel(self, modelpath):
        self.DeepNet = load_model(modelpath)

    def Save(self, modelpath):
        self.DeepNet.save(modelpath)

    def GetModel(self):
        return self.DeepNet


Datapath="StateActionData.txt"
Labelpath="StateActionLabel.txt"
DNN_DELTA=FullyConnectedDeepNet(6,100,2,4,0.1)
DNN_INJAIL=FullyConnectedDeepNet(6,100,2,1,0.1)
Full_Data=ReadData(Datapath)
Full_Label=ReadData(Labelpath)
Train_Data = Full_Data[:90000]
Train_Label = Full_Label[:90000]
Test_Data = Full_Data[90000:]
Test_Label = Full_Label[90000:]
DNN_DELTA.Train(Train_Data,Train_Label[:,:2],30)
DNN_INJAIL.Train(Train_Data,Train_Label[:,2:],30)
DNN_DELTA.Save("TransitionModel_DELTA.h5")
DNN_INJAIL.Save("TransitionModel_INJAIL.h5")


Datapath="StateActionData.txt"
Labelpath="StateActionLabel.txt"
Full_Data=ReadData(Datapath)
Full_Label=ReadData(Labelpath)
Train_Data = Full_Data[:90000]
Train_Label = Full_Label[:90000]
Test_Data = Full_Data[90000:]
Test_Label = Full_Label[90000:]
DNN_DELTA=FullyConnectedDeepNet(6,100,2,4,0.1)
DNN_INJAIL=FullyConnectedDeepNet(6,100,2,1,0.1)
DNN_DELTA.LoadModel("TransitionModel_DELTA.h5")
DNN_INJAIL.LoadModel("TransitionModel_INJAIL.h5")

Test_Pred = np.concatenate((DNN_DELTA.Test(Test_Data), DNN_INJAIL.Test(Test_Data)), axis=1)
print(Test_Pred[:3])
print(Test_Label[:3])


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def Virtualizing(Data, Label, Pred, Obstacles,SampleSize):
    sample_index=np.random.choice(len(Data), SampleSize)
    #print sample_index
    #print Train_Data[sample_index,:2],Train_Label[sample_index,:2]
    fig9 = plt.figure()
    ax9 = fig9.add_subplot(111)
    for i in sample_index:
        plt.plot([Data[i,0],Data[i,0]+Label[i,0]], [Data[i,1],Data[i,1]+Label[i,1]], 'r-',lw=1)
        plt.plot([Data[i,0],Data[i,0]+Pred[i,0]], [Data[i,1],Data[i,1]+Pred[i,1]], 'k-',lw=1)
    for j in range(len(Obstacles)):
        ax9.add_patch(patches.Rectangle((Obstacles[j][0], Obstacles[j][1]),
                                        Obstacles[j][2]-Obstacles[j][0], Obstacles[j][3]-Obstacles[j][1],
                                        hatch='\\', fill=False,linestyle='solid'))
    plt.show()

Virtualizing(Test_Data,Test_Label,Test_Pred,Maze.obstacles,500)


DNN_DELTA=FullyConnectedDeepNet(6,100,2,4,0.1)
# Train_Data=ReadData(Datapath)
# Train_Label=ReadData(Labelpath)
DNN_DELTA.LoadModel("TransitionModel_DELTA.h5")
for layer in DNN_DELTA.GetModel().layers:
    g=layer.get_config()
    h=layer.get_weights()
    print(g)
    print(h)


class NetTopology(object):
    def __init__(self, layers, drop_out):
        self.layers = layers
        self.input_dim = layers[0].get_config()[0].get('config').get('input_dim')
        self.num_upper_layers = len(layers)
        self.nodenames = []
        self.drop_out = drop_out
        layernodename = []
        for i in range(0, self.input_dim):
            layernodename.append('N0' + str(i))
        layernodename.append('B0')
        self.nodenames.append(layernodename)

    def LayerWiseTransform(self, layer, layer_id, lastlayer=False, hiddenstart='N', writefile=False, filehandler=None):
        if not lastlayer:
            input_dim = layer.get_config()[0].get('config').get('input_dim')
            output_dim = layer.get_config()[0].get('config').get('output_dim')
            activation = layer.get_config()[1].get('config').get('activation')
        else:
            input_dim = layer.get_config().get('input_dim')
            output_dim = layer.get_config().get('output_dim')
            activation = layer.get_config().get('activation')
        layernodename = []
        weights_bias = layer.get_weights()
        weights = weights_bias[0]
        bias = weights_bias[1]
        for i in range(0, output_dim):
            layernodename.append(hiddenstart + str(layer_id) + str(i))
        for i in range(0, output_dim):
            row = [layernodename[i], activation]
            for j in range(0, input_dim):
                row.append(self.nodenames[-1][j])
                row.append(weights[j][i] * self.drop_out)
            row.append(self.nodenames[-1][-1])
            row.append(bias[i])
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
            if type(self.layers[i]) is not Dropout:
                if (i == self.num_upper_layers - 1):
                    self.LayerWiseTransform(self.layers[i], counter + 1, True, hiddenstart, writefile, filehandler)
                else:
                    self.LayerWiseTransform(self.layers[i], counter + 1, False, hiddenstart, writefile, filehandler)
                counter = counter + 1
        if writefile:
            filehandler.close()
        print('Done!')


Topo_DELTA = NetTopology(DNN_DELTA.GetModel().layers, DNN_DELTA.drop_out)
Topo_DELTA.NetTransform('D', True, "Network_MIX.txt", True)

Topo_INJAIL = NetTopology(DNN_INJAIL.GetModel().layers, DNN_INJAIL.drop_out)
Topo_INJAIL.NetTransform('I', True, "Network_MIX.txt", False)


