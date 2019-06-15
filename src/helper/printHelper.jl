using Printf
using DelimitedFiles
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

function printConvDecLayers(cnn, name, val)
  println("$name: ")
  for k in 1:cnn.numConv
    println("Layer ", k, ':')
    for c in 1:cnn.channels[k]
      for i in 1:cnn.layerSizes[k]
        for j in 1:cnn.layerSizes[k]
          print("$(@sprintf("%.2f",getvalue(val[k,c,i,j]))) ")
        end
        println(" ")
      end
      println(" ")
    end
  end
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
function printVars(nn::NeuralNet, x, s, predLabel, label, printWeights, xRange=2:5, wRange=1:4)

  println(" ")
  println("Objective Value: ", getobjectivevalue(nn.m))
  println("Solve Time: ", getsolvetime(nn.m))
  println(" ")

  sizes = nn.layerSizes

  printDecLayers("X", x, xRange, sizes)
  printDecLayers("S", s, xRange, sizes)
  if printWeights
    printLayers("Weights", nn.w, wRange, sizes)
    printLayers("Biases", nn.b, wRange, sizes)
  end
  println("====================================")
  modelPredLabel = getPredictedLabel(nn.m,x,sizes)
  println("True label: $(label)")
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
function writeResultToCSV(path, csvData)
    csvFileName = replace(path, "datasets/" => "")
    csvFileName = replace(csvFileName, ".json" => "")
    csvFileName = replace(csvFileName, "/" => "")
    writedlm("$csvFileName-adversarial-mipfocus=2.csv", csvData)
end