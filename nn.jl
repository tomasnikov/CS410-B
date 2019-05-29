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
function modelNN(input, w, b, label, layerSizes)
  
  m = Model(solver=GurobiSolver())

  numLayers = size(layerSizes,1)

  # Initialize x decision variable, x >= 0. x[k,j] is x^k_j. Arbitrary upper bound of 10000
  @variable(m, 0 <= x[k=1:numLayers, j=1:layerSizes[k]] <= 10000)
  # Initialize s decision variable, s >= 0. s[k,j] is s^k_j
  @variable(m, 0 <= s[k=2:numLayers, j=1:layerSizes[k]] <= 10000)
  # Initialize binary z decision variable
  @variable(m, z[k=2:numLayers, j=1:layerSizes[k]], Bin)

  # First x layer must be equal to the input
  @constraint(m, [x[1,j] for j in 1:layerSizes[1]] .== input)

  lastLayer = [x[1,j] for j in 1:layerSizes[1]]
  for k in 2:numLayers
    # x_j, s_j and z_j are values for layer k, j in 1:n_k
    layerX = [x[k,j] for j in 1:layerSizes[k]]
    layerS = [s[k,j] for j in 1:layerSizes[k]]
    layerZ = [z[k,j] for j in 1:layerSizes[k]]

    weights = w[string(k-1)]
    biases = b[string(k-1)]

    left = transpose(weights) * lastLayer + biases

    # ReLU formulation, w^{k-1}^T * x^{k-1} + b^{k-1} = x^k - s^k
    @constraint(m, left .== layerX .- layerS)

    for j in 1:layerSizes[k]
      @disjunction(m, (x[k,j] == 0), (s[k,j] == 0))
    end

    lastLayer = layerX
  end
  
  # Constrain output to be equal to expected output
  output = [x[numLayers,j] for j in 1:layerSizes[numLayers,]]
  for k in 1:layerSizes[numLayers]
    if k != label+1
      @constraint(m, x[numLayers,label+1] >= x[numLayers,k])
    end
  end

  # Minimize objective function
  @objective(m, Min, sum(x))
  #@objective(m, Min, sum(x .* c) + sum(z .* g))

  print("Solver time: $(@elapsed(solve(m)))"
  return m,x,s,z,output
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
        print(tmp[i,j], " ")
      end
      println(" ")
    end
  end
  println("-------------")
end

"""
Print all variables
"""
function printVars(m,x,s,z,w,b,output,layerSizes,label,printWeights)

  numLayers = size(layerSizes,1)
  
  println(" ")
  println("Objective Value: ", getobjectivevalue(m))
  println(" ")

  printDecLayers("X", x, 2:numLayers, layerSizes)
  printDecLayers("S", s, 2:numLayers, layerSizes)
  printDecLayers("Z", z, 2:numLayers, layerSizes)
  if printWeights
    printLayers("Weights", w, 1:numLayers-1, layerSizes)
    printLayers("Biases", b, 1:numLayers-1, layerSizes)
  end
  println("====================================")
  maxVal = findmax([getvalue(x[numLayers,j]) for j in 1:layerSizes[numLayers]])
  maxInd = maxVal[2] - 1
  println("Target label: $label")
  println("Classification: $maxInd")
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
  return layers,input, label, w, b
end

function main()
  # Read JSON file
  file = ARGS[1]

  # Load data from file
  layers,input,label,w,b = loadData(file)

  println("Now constraining!")
  t = @elapsed m,x,s,z,output = modelNN(input,w,b,label,layers)
  println(" ")
  println("Time: ",t)
  printVars(m,x,s,z,w,b,output,layers,label,false)
end


main()