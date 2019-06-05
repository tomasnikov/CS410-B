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
  println("Solve Time: ", getsolvetime(nn.m))
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

function printResults(numImages::Int64, sameAsNN, sameAsTrue, nnSameAsTrue)
  println("Total number of instances: $numImages")
  println("Number of model prediction equal to NN prediction: $sameAsNN")
  println("Number of model prediction equal to True labels: $sameAsTrue")
  println("Number of NN prediction equal to True labels: $nnSameAsTrue")
end

function classificationCheck(predLabel, modelPredLabel, trueLabel)
    return predLabel == modelPredLabel, trueLabel == modelPredLabel, trueLabel == predLabel
end


"""
    Write input to file
"""
function writeInputToJSON(input, targetLabel, modelPredLabel, filename)
  data = Dict("input" => input, "label" => targetLabel, "predictedLabel" => modelPredLabel)
  json_data = JSON.json(data)
  open(filename, "w") do f
    write(f, json_data)
  end
end

"""
    Write to csv
"""
function writeResultToCSV(path)
    csvFileName = replace(path, "datasets/" => "")
    csvFileName = replace(csvFileName, ".json" => "")
    csvFileName = replace(csvFileName, "/" => "")
    writedlm("$csvFileName-adversarial.csv", csvData)
end