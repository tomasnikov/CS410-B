#=
problemLoader:
- Julia version: 1.1.0
- Author: veerle
- Date: 2019-06-03
=#
using Printf

import JSON

"""
Load data from json file 'file'
"""
function loadData(file)
  f = open(file)
  data = JSON.parse(String(read(f)))
  layers = Int64.(data["layers"])
  println(layers)
  input = Float64.(data["input"])
  label = data["label"]
  println("Label: $label")
  predLabel = data["predictedLabel"]
  println("NN Predicted Label: $predLabel")
  numLayers  = size(layers,1)

  # Create weight dict, w[k][i,j] is weight i,j in layer k
  w = Dict(string(k) => zeros(layers[k],layers[k+1]) for k in 1:numLayers-1)
  # Create bias dict, b[k][j] is bias j in layer k
  b = Dict(string(k) => zeros(layers[k+1],1) for k in 1:numLayers-1)

  # Load weight data
  for k in 1:numLayers-1
    weights = data["weights"][string(k)]
    for (i,row) in enumerate(weights)
      for (j,val) in enumerate(row)
        w[string(k)][i,j] = val
      end
    end
  end

  # Load bias data
  for k in 1:numLayers-1
    biases = data["biases"][string(k)]
    for (i,row) in enumerate(biases)
      for (j,val) in enumerate(row)
        b[string(k)][i,j] = val
      end
    end
  end
  return layers, input, label, predLabel, w, b
end