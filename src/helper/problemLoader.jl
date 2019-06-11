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

"""
Load CNN data from json file 'file
"""
function loadCNNData(file)
  f = open(file)
  data = JSON.parse(String(read(f)))
  layers = Int64.(data["layers"])
  channels = [1,3,3,1,1]
  padding = 1
  if occursin("convcnn2", file)
    channels = [1, 6, 6, 16, 16, 1, 1] # Not sure about this
  end
  input = Float64.(data["input"])
  # Original input is flat (1,784) so must reshape and transpose
  input = transpose(reshape(input, (28,28)))

  label = data["label"]
  predLabel = data["predictedLabel"]

  numConv = length(keys(data["weights"]["conv"]))
  numFc = length(keys(data["weights"]["fc"]))

  # Creates conv weights/biases i.e. {"1": {"1": [[w11, w12, w13],[w21,w22,w23],[w31,w32,w33]]}} etc.
  convW = Dict(string(k) => Dict(string(c) => zeros(3,3) for c in 1:size(data["weights"]["conv"][string(k)],1)) for k in 1:numConv)
  convB = Dict(string(k) => zeros(3,1) for k in 1:numConv)

  # Create fc weights/biases consistent with current layer, i.e. {"3": ...}.
  # The +1 is for the offset of pooling
  w = Dict(string(numConv+k+1) => zeros(layers[k+2],layers[k+3]) for k in 1:numFc)
  b = Dict(string(numConv+k+1) => zeros(layers[k+3],1) for k in 1:numFc)
  
  for k in 1:numConv
    convWeights = data["weights"]["conv"][string(k)]
    convBiases = data["biases"]["conv"][string(k)]
    for c in 1:size(convWeights,1)
      channelConvWeights = convWeights[c]
      convB[string(k)][c,1] = convBiases[c]
      for (i,row) in enumerate(channelConvWeights)
        for (j,val) in enumerate(row)
          convW[string(k)][string(c)][i,j] = val
        end
      end
    end
  end

  for k in 1:numFc
    fcWeights = data["weights"]["fc"][string(k)]
    for (i,row) in enumerate(fcWeights)
      for (j,val) in enumerate(row)
        w[string(numConv+k+1)][i,j] = val
      end
    end
    fcBiases = data["biases"]["fc"][string(k)]
    for (i,row) in enumerate(fcBiases)
      for (j,val) in enumerate(row)
        b[string(numConv+k+1)][i,j] = val
      end
    end
  end

  # Add for input and multiply for pooling layers
  numConv = numConv*2 + 1

  layers = layers ./ channels
  layers[1:numConv] = [Int(sqrt(c)) for c in layers[1:numConv]]
  layers[1] += 2*padding
  layers = Int.(layers)

  return layers, input, label, predLabel, convW, w, convB, b, channels, numConv
end


"""
 Based on the command line arguments (ARGS) determine what actions are required and which file(s) to read
"""
function processArgs(args)
     # Set to true if output label is hard constraint (i.e. adversarial)
  doAdversarial = false
  # Set to true to write adversarial to JSON
  writeToJSON = false
  # Set to true to see all the weights when printing
  printWeights = false
  # Set to true to write to csv
  writeToCSV = false

  if "--adversarial" in args
    doAdversarial = true
  end
  if "--writeJSON" in args
    writeToJSON = true
  end
  if "--writeCSV" in args
    writeToCSV = true
  end
  if "--printWeights" in args
    printWeights = true
  end

  # Read JSON file
  path = ARGS[1]
  if occursin(".json", path)
    files = [path]
  else
    files = ["$path$f" for f in readdir(path)]
  end

  return files, doAdversarial, writeToJSON, writeToCSV, printWeights
end