using JuMP
using Gurobi

include("helper/printHelper.jl")
include("helper/problemLoader.jl")
include("helper/modelHelper.jl")
include("NeuralNet.jl")


function main()
  files, doAdversarial, writeToJSON, printWeights = processArgs(ARGS)
  sameAsCNN = sameAsTrue = CNNequalToTrue = false

  numImages = length(files)
  for file in files
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
    m = Model(solver=GurobiSolver())
    cnn::CNN = CNN(m, input, convW, w, convB, b, layers, targetLabel, channels,numConv)

    # build and solve the model
    t = @elapsed m,convX,fcX,convS,fcS = cnn(doAdversarial)
    println("\n\nTotal runtime: ",t)

    # Change me if you want a huge console log!
    if true
      printConvDecLayers(cnn, "ConvX", convX)
    end

    println("-------------")
    numLayers = size(cnn.layerSizes,1)

    printVars(cnn, fcX, fcS, predLabel, label, printWeights, numConv+1:numLayers)
    modelPredLabel = getPredictedLabel(m,fcX,layers)

    sameAsCNN, sameAsTrue, CNNequalToTrue = classificationCheck(predLabel, modelPredLabel, label)

    if doAdversarial && writeToJSON
      new_input = [getvalue(x[1,j]) for j in 1:layers[1]]
      fileName = match(r"/\d.*.json", file).match
      writeInputToJSON(new_input, cnn.targetLabel, modelPredLabel, "adversarials/cnn1$fileName")
    end
    
  end
  printResults(numImages,sameAsCNN, sameAsTrue, CNNequalToTrue)
end

main()