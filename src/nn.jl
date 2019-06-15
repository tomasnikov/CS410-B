using JuMP
using Gurobi

include("helper/printHelper.jl")
include("helper/problemLoader.jl")
include("helper/modelHelper.jl")
include("NeuralNet.jl")


function main()
  files, doAdversarial, writeToJSON, writeToCSV, printWeights = processArgs(ARGS)
  sameAsNN = sameAsTrue = NNequalToTrue = false

  numImages = length(files)

  csvData = zeros((numImages, 8))
  counter = 0

  for file in files
    println(file)
    counter += 1
    # Load data from file
    layers,input,label,predLabel,w,b = loadData(file)
    numLayers = size(layers,1)
    # Normalize input
    input = input/findmax(input)[1]

    targetLabel = label
    if doAdversarial
      targetLabel = (label + 5) % 10
    end

    println("Now constraining!")

    # Initialize model and NeuralNet
    m = Model(solver=GurobiSolver(TimeLimit = 300, MIPFocus=2))
    nn::NN = NN(m, input, w, b, layers, targetLabel)

    # build and solve the model
    t = @elapsed m,x,s = nn(doAdversarial)
    println("\n\nTotal runtime: ",t)
    printVars(nn, x, s, predLabel, label, printWeights, 2:numLayers, 1:numLayers-1)
    modelPredLabel = getPredictedLabel(m,x,layers)

    sameAsNN, sameAsTrue, NNequalToTrue = classificationCheck(predLabel, modelPredLabel, label)

    if doAdversarial && writeToJSON
      new_input = [getvalue(x[1,j]) for j in 1:layers[1]]
      advFileName = replace(file, "datasets" => "adversarials")
      writeInputToJSON(new_input, targetLabel, modelPredLabel, advFileName)
    end

    value,solvetime,bound,nodes,gap = getRunResults(m)
    csvData[counter,:] = [label, predLabel, modelPredLabel, value, bound, gap, solvetime, nodes]
    
  end

  if writeToCSV
    writeResultToCSV(ARGS[1], csvData)
  end

  printResults(numImages,sameAsNN, sameAsTrue, NNequalToTrue)
end

main()