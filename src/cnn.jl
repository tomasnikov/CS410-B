using JuMP
using Gurobi

include("helper/printHelper.jl")
include("helper/problemLoader.jl")
include("helper/modelHelper.jl")
include("NeuralNet.jl")


function main()
  files, doAdversarial, writeToJSON, writeToCSV, printWeights = processArgs(ARGS)
  sameAsCNN = sameAsTrue = CNNequalToTrue = false

  numImages = length(files)
  csvData = zeros((numImages, 8))
  counter = 0

  for file in files
      counter += 1
    # Load data from file
    layers,input,label,predLabel,convW,w,convB,b,channels,numConv = loadCNNData(file)
    # Normalize input
    input = input/findmax(input)[1]

    targetLabel = label
    if doAdversarial
      targetLabel = (label + 5) % 10
    end

    println("Now constraining!")

    # Initialize model and NeuralNet
    m = Model(solver=GurobiSolver(TimeLimit = 60))
    cnn::CNN = CNN(m, input, convW, w, convB, b, layers, targetLabel, channels,numConv)

    # build and solve the model
    t = @elapsed m,convX,fcX,convS,fcS = cnn(doAdversarial)
    println("\n\nTotal runtime: ",t)

    # Change me if you want a huge console log!
    if false
      printConvDecLayers(cnn, "ConvX", convX)
    end

    println("-------------")
    numLayers = size(cnn.layerSizes,1)

    printVars(cnn, fcX, fcS, predLabel, label, printWeights, numConv+1:numLayers)
    modelPredLabel = getPredictedLabel(m,fcX,layers)

    sameAsCNN, sameAsTrue, CNNequalToTrue = classificationCheck(predLabel, modelPredLabel, label)

    if doAdversarial && writeToJSON
      new_input = [getvalue(convX[1,1,i,j]) for i = 2:layers[1]-1, j = 2:layers[1]-1]
      # Have to transpose for some reason
      new_input = transpose(new_input)
      fileName = match(r"/\d.*.json", file).match
      writeInputToJSON(new_input, cnn.targetLabel, modelPredLabel, "adversarials/convnn1$fileName")
    end

    value,solvetime,bound,nodes,gap = getRunResults(m)
    csvData[counter,:] = [label, predLabel, modelPredLabel, value, bound, gap, solvetime, nodes]

  end

  if writeToCSV
    writeResultToCSV(ARGS[1], csvData)
  end

  printResults(numImages,sameAsCNN, sameAsTrue, CNNequalToTrue)
end

main()