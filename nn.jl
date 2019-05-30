using JuMP
using Gurobi
using Distributions
using LinearAlgebra
using DelimitedFiles
using Printf
using ConditionalJuMP

import JSON

"""
Add constraints for neural network, then optimize.
"""
function modelNN(input, w, b, label, layerSizes, doAdversarial)
  
  m = Model(solver=GurobiSolver())

  numLayers = size(layerSizes,1)

  # Initialize x decision variable, x >= 0. x[k,j] is x^k_j. Arbitrary upper bound of 10000
  @variable(m, 0 <= x[k=1:numLayers, j=1:layerSizes[k]] <= 10000)
  # Initialize s decision variable, s >= 0. s[k,j] is s^k_j
  @variable(m, 0 <= s[k=2:numLayers, j=1:layerSizes[k]] <= 10000)

  if doAdversarial
    # Introduce d variable, represents distance between input and adversarial input
    @variable(m, 0 <= d[1,j=1:layerSizes[1]] <= 10000)

    firstLayerX = [x[1,j] for j in 1:layerSizes[1]]
    firstLayerD = [d[1,j] for j in 1:layerSizes[1]]

    # Input must be in range 0 to 1
    @constraint(m, firstLayerX .<= .1)
    # Constrain d to be distance between original input and x input
    @constraint(m, -firstLayerD .<= firstLayerX - input)
    @constraint(m, firstLayerD .>= firstLayerX - input)
  else
    # First x layer must be equal to the input
    @constraint(m, [x[1,j] for j in 1:layerSizes[1]] .== input)
  end

  for k in 2:numLayers
    # x_j and s_j are values for layer k, j in 1:n_k
    layerX = [x[k,j] for j in 1:layerSizes[k]]
    layerS = [s[k,j] for j in 1:layerSizes[k]]

    weights = w[string(k-1)]
    biases = b[string(k-1)]
    lastLayer = [x[k-1,j] for j in 1:layerSizes[k-1]]

    # ReLU formulation, w^{k-1}^T * x^{k-1} + b^{k-1} = x^k - s^k
    @constraint(m, transpose(weights) * lastLayer + biases .== layerX .- layerS)

    # Either s or x must be 0
    for j in 1:layerSizes[k]
      @disjunction(m, (x[k,j] == 0), (s[k,j] == 0))
    end
  end

  # Only do the following if output has a hard constraint (i.e. adversarial)
  if doAdversarial
    # Constrain output to be equal to expected output
    output = [x[numLayers,j] for j in 1:layerSizes[numLayers,]]
    for k in 1:layerSizes[numLayers]
      if k != label+1
        @constraint(m, x[numLayers,label+1] >= 1.2*x[numLayers,k])
      end
    end
  end

  if doAdversarial
    # Minimize distance between original input and adversarial input
    @objective(m, Min, sum(d))
  else
    # Minimize over all x. Since input is fixed, this does not really do anything
    @objective(m, Min, sum(x))
  end

  print("Solver time: $(@elapsed(solve(m)))")
  return m,x,s
end

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

function writeInputToJSON(input, targetLabel, modelPredLabel, filename)
  data = Dict("input" => input, "label" => targetLabel, "predictedLabel" => modelPredLabel)
  json_data = JSON.json(data)
  open(filename, "w") do f
    write(f, json_data)
  end
end

function main()
  # Read JSON file
  path = ARGS[1]
  if occursin(".json", path)
    files = [path]
  else
    files = ["$path$f" for f in readdir(path)]
  end
  # Set to true if output label is hard constraint (i.e. adversarial)
  doAdversarial = true
  # Set to true to write adversarial to JSON
  writeToJSON = true
  # Set to true to see all the weights when printing
  printWeights = false

  sameAsNN = 0
  sameAsTrue = 0
  NNequalToTrue = 0
  numImages = length(files)
  for file in files
    # Load data from file
    layers,input,label,predLabel,w,b = loadData(file)
    # Normalize input
    input = input/findmax(input)[1]

    targetLabel = label
    if doAdversarial
      targetLabel = (label + 5) % 10
    end

    println("Now constraining!")
    t = @elapsed m,x,s = modelNN(input,w,b,targetLabel,layers,doAdversarial)
    println(" ")
    println("Time: ",t)
    printVars(m,x,s,w,b,layers,label,predLabel,printWeights)
    modelPredLabel = getPredictedLabel(m,x,layers)

    if predLabel == modelPredLabel
      sameAsNN += 1
    end
    if label == modelPredLabel
      sameAsTrue += 1
    end
    if label == predLabel
      NNequalToTrue +=1
    end

    if doAdversarial && writeToJSON
      new_input = [getvalue(x[1,j]) for j in 1:layers[1]]
      fileName = match(r"/\d.*.json", file).match
      writeInputToJSON(new_input, targetLabel, modelPredLabel, "adversarials/nn1$fileName")
    end
    
  end
  println("Total number of instances: $numImages")
  println("Number of labels equal to NN prediction: $sameAsNN")
  println("Number of labels equal to True labels: $sameAsTrue")
  println("Number of NN predictions equal to True labels: $NNequalToTrue")
  
end


main()