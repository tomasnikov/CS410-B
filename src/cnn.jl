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
    layers,input,label,predLabel,convW,fcW,convB,fcB,channels = loadCNNData(file)
    # Normalize input
    input = input/findmax(input)[1]

    targetLabel = label
    if doAdversarial
      targetLabel = (label + 5) % 10
    end

    println("Now constraining!")

    # Initialize model and NeuralNet
    m = Model(solver=GurobiSolver())
    cnn::CNN = CNN(m, input, convW, fcW, convB, fcB, layers, targetLabel, channels)

    # build and solve the model
    t = @elapsed m,convX,fcX = cnn(doAdversarial)
    println("\n\nTotal runtime: ",t)
    println("x Values: ")
    totalSum = 0
    for k in 1:3
      println("Layer ", k, ':')
      padding = 0
      if k == 1
        padding = 2
      end
      for c in 1:channels[k]
        for i in 1:Int(sqrt(layers[k]/channels[k]))+padding
          for j in 1:Int(sqrt(layers[k]/channels[k]))+padding
            print("$(@sprintf("%.2f",getvalue(convX[k,c,i,j]))) ")
            totalSum += getvalue(convX[k,c,i,j])
          end
          println(" ")
        end
        println(" ")
      end
    end
    println("-------------")
    numLayers = size(cnn.layerSizes,1)

    printDecLayers("fcX", fcX, 4:numLayers, layers)
    println(totalSum)
    #printVars(cnn, x,s,predLabel,printWeights)
    #modelPredLabel = getPredictedLabel(m,x,layers)

    #sameAsCNN, sameAsTrue, CNNequalToTrue = classificationCheck(predLabel, modelPredLabel, label)

    if doAdversarial && writeToJSON
      new_input = [getvalue(x[1,j]) for j in 1:layers[1]]
      fileName = match(r"/\d.*.json", file).match
      writeInputToJSON(new_input, cnn.targetLabel, modelPredLabel, "adversarials/cnn1$fileName")
    end
    
  end
  printResults(numImages,sameAsCNN, sameAsTrue, CNNequalToTrue)
end

main()