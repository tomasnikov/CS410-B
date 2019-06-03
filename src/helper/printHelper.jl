using Printf

"""
Print decision variable 'val' with name 'name'
"""
function printDecLayers(name, val, range, layerSizes)
  println("$name Values: ")
  for k in range
    println("Layer ", k, ':')
    for j in 1:layerSizes[k]
      print("$(@sprintf("%.2f",getvalue(val[k,j]))) ")
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
        print("$(@sprintf("%.2f",tmp[i,j])) ")
      end
      println(" ")
    end
  end
  println("-------------")
end


function getPredictedLabel(m,x,layerSizes)
  numLayers = size(layerSizes,1)
  maxVal = findmax([getvalue(x[numLayers,j]) for j in 1:layerSizes[numLayers]])
  label = maxVal[2] - 1
  return label
end

"""
Print all variables
"""
function printVars(m,x,s,w,b,layerSizes,label,predLabel,printWeights)

  numLayers = size(layerSizes,1)

  println(" ")
  println("Objective Value: ", getobjectivevalue(m))
  println("Time: ", getsolvetime(m))
  println(" ")

  printDecLayers("X", x, 2:numLayers, layerSizes)
  printDecLayers("S", s, 2:numLayers, layerSizes)
  if printWeights
    printLayers("Weights", w, 1:numLayers-1, layerSizes)
    printLayers("Biases", b, 1:numLayers-1, layerSizes)
  end
  println("====================================")
  modelPredLabel = getPredictedLabel(m,x,layerSizes)
  println("True label: $label")
  println("NN Predicted Label: $predLabel")
  println("Model Predicted Label: $modelPredLabel")
  println(" ")
end