from NNet.converters.onnx2nnet import onnx2nnet


# ONNX file to convert to .nnet file
onnxFile = 'model.onnx'

# Convert the file
onnx2nnet(onnxFile)