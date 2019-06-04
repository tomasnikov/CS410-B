using Printf
include("../NeuralNet.jl")

"""
Print decision variable 'val' with name 'name'
"""
function printDecLayers(name, val, range, layerSizes)
  println("$name Values: ")
  for k in range
    println("Layer ", k, ':')
    for j in 1:layerSizes[k]
      print("$(@sprintf("%.2f",getvalue(val[k, j]))) ")
    end
    println(" ")
  end
  println("-------------")
end

"""
Print regular variable 'val' with name 'name'
"""
function printLayers(name, val, range, layerSizes)
  println("$name: ")
  for k in range
    println("Layer ", k, ':')
    tmp = val[string(k)]
    for i in 1:size(tmp,1)
      for j in 1:size(tmp,2)
        print("$(@sprintf("%.2f",tmp[i, j])) ")
      end
      println(" ")
    end
  end
  println("-------------")
end


function getPredictedLabel(m, x, layerSizes)
  numLayers = size(layerSizes,1)
  maxVal = findmax([getvalue(x[numLayers, j]) for j in 1:layerSizes[numLayers]])
  label = maxVal[2] - 1
  return label
end

"""
Print all variables
"""
function printVars(nn::NeuralNet, x, s, predLabel, printWeights)

  numLayers = size(nn.layerSizes,1)

  println(" ")
  println("Objective Value: ", getobjectivevalue(nn.m))
  println("Time: ", getsolvetime(nn.m))
  println(" ")

  sizes = nn.layerSizes

  printDecLayers("X", x, 2:numLayers, sizes)
  printDecLayers("S", s, 2:numLayers, sizes)
  if printWeights
    printLayers("Weights", nn.w, 1:numLayers - 1, sizes)
    printLayers("Biases", nn.b, 1:numLayers - 1, sizes)
  end
  println("====================================")
  modelPredLabel = getPredictedLabel(nn.m,x,sizes)
  println("True label: $(nn.targetLabel)")
  println("NN Predicted Label: $predLabel")
  println("Model Predicted Label: $modelPredLabel")
  println(" ")
end