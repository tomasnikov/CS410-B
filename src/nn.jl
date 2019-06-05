using JuMP
using Gurobi

include("helper/printHelper.jl")
include("helper/problemLoader.jl")
include("helper/modelHelper.jl")
include("NeuralNet.jl")


function getRunResults(m)
  value = getobjectivevalue(m)
  solvetime = getsolvetime(m)
  bound = getobjectivebound(m)
  nodes = getnodecount(m)
  gap = abs(value - bound)/value
  return value,solvetime,bound,nodes,gap
end

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
    # Normalize input
    input = input/findmax(input)[1]

    targetLabel = label
    if doAdversarial
      targetLabel = (label + 5) % 10
    end

    println("Now constraining!")

    # Initialize model and NeuralNet
    m = Model(solver=GurobiSolver())
    nn::NN = NN(m, input, w, b, layers, targetLabel)

    # build and solve the model
    t = @elapsed m,x,s = nn(doAdversarial)
    println("\n\nTotal runtime: ",t)
    printVars(nn, x,s,predLabel,printWeights)
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
    writeResultToCSV(ARGS[1])
  end

  printResults(numImages,sameAsNN, sameAsTrue, NNequalToTrue)
end

main()