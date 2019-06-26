
from DCN_Transition_Function_RDDL import NetTopology, DenselyConnectedNetwork

DNN = DenselyConnectedNetwork(6,100,2,4,0.1)
DNN.DeepNet.load('HVAC.h5')

for layer in DNN.GetModel().layers:
    g=layer.get_config()
    h=layer.get_weights()
    print(g)
    print(h)


# MILP encoding of deep neural net
Topo_DELTA = NetTopology(DNN.GetModel().layers, DNN.drop_out)
Topo_DELTA.NetTransform('D', True, "Network_MIX.txt", True)