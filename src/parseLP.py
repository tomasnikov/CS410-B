from pdb import set_trace
import re
import numpy as np
import json
from mlxtend.data import loadlocal_mnist

W1 = np.zeros((784,8))
W2 = np.zeros((8,8))
W3 = np.zeros((8,8))
W4 = np.zeros((8,10))

B1 = np.zeros(8)
B2 = np.zeros(8)
B3 = np.zeros(8)
B4 = np.zeros(10)

allWeights = [W1,W2,W3,W4]
allBiases = [B1,B2,B3,B4]

offsets = [3,11,19,27,35,45]
ranges = [range(3,11), range(11,19), range(19,27), range(27,37)]

file = 'fichetti_data/dnn1_1sec/dnn1_1sec_0.lp'

data = open(file, 'r').read().split(':')

for k,layer in enumerate(allWeights):
  for j,neuron in enumerate(ranges[k]):
    w = data[neuron]

    w = re.sub(r"x\(\d*,\d*\)", ',', w)
    w = re.sub(r"s\(\d*,\d*\)", '', w)
    w = w.replace('\n', '').replace(' ', '').replace('+','')
    w = w.split(',')
    b = re.sub(r"unit.*",'',w[-1])
    b = re.sub(r"delta.*",'',b)
    b = b.replace('=','')
    b = float(b)

    w = w[0:-2]
    w = [float(tmpW) for  tmpW in w]
    allWeights[k][:,j] = w
    allBiases[k][j] = b

X, y = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')

newdata = {
  "layers": [784,8,8,8,10],
  "input": X[0].tolist(),
  "label": int(y[0]),
  "weights": {
    "1": W1.tolist(),
    "2": W2.tolist(),
    "3": W3.tolist(),
    "4": W4.tolist(),
  },
  "biases": {
    "1": B1.tolist(),
    "2": B2.tolist(),
    "3": B3.tolist(),
    "4": B4.tolist(),
  }
}

with open('datasets/tmp.json', 'w') as outfile:
  json.dump(newdata, outfile)