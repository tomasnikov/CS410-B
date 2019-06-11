#=
modelHelper:
- Julia version: 1.1.0
- Author: veerle
- Date: 2019-06-04
=#
using JuMP, ConditionalJuMP
include("../NeuralNet.jl")

"""
Add the relu and binary constraint
"""
function add_relu_constraints(nn::NeuralNet, start, final, x, s, firstLayer)
  lastLayer = firstLayer
  for k in start:final
    # x_j and s_j are values for layer k, j in 1:n_k
    layerX = [x[k,j] for j in 1:nn.layerSizes[k]]
    layerS = [s[k,j] for j in 1:nn.layerSizes[k]]

    weights = nn.w[string(k-1)]
    biases = nn.b[string(k-1)]

    # ReLU formulation, w^{k-1}^T * x^{k-1} + b^{k-1} = x^k - s^k
    @constraint(nn.m, transpose(weights) * lastLayer + biases .== layerX .- layerS)

    # Either s or x must be 0
    for j in 1:nn.layerSizes[k]
      @disjunction(nn.m, (x[k,j] == 0), (s[k,j] == 0))
    end
    lastLayer = layerX
  end
end

"""
    Introduce d variable, represents distance between input and adversarial input
"""
function addAdversarialConstraints(nn, x, firstLayerX, firstLayerD)
    numLayers = size(nn.layerSizes,1)

    # Input must be in range 0 to 1
    @constraint(nn.m, firstLayerX .<= 1)
    # Constrain d to be distance between original input and x input
    @constraint(nn.m, -firstLayerD .<= firstLayerX - nn.input)
    @constraint(nn.m, firstLayerD .>= firstLayerX - nn.input)

    # Constrain output to be equal to expected output
    output = [x[numLayers, j] for j in 1:nn.layerSizes[numLayers, ]]
    label = nn.targetLabel
    for k in 1:nn.layerSizes[numLayers]
      if k != label+1
        @constraint(nn.m, x[numLayers,label+1] >= 1.2*x[numLayers,k])
      end
    end
end

"""
    Constrain all padded values to be equal to zeros
"""
function addPaddingConstraint(cnn, convX, inputSize)
  m = cnn.m
  @constraint(m, [convX[1,1,1,        j] for j=1:inputSize] .== 0)
  @constraint(m, [convX[1,1,inputSize,j] for j=1:inputSize] .== 0)
  @constraint(m, [convX[1,1,i,        1] for i=1:inputSize] .== 0)
  @constraint(m, [convX[1,1,i,inputSize] for i=1:inputSize] .== 0)
end

function addAveragePoolConstraint(cnn, convX, layerNum)
  m = cnn.m
  numChannels = cnn.channels[layerNum]
  convSize = cnn.layerSizes[layerNum]
  for c = 1:numChannels
    for i = 1:convSize
      for j = 1:convSize
        region = [convX[2,c,a,b] for a=2*i-1:2*i,b=2*j-1:2*j]
        @constraint(m, convX[3,c,i,j] == sum(region)/4)
      end
    end
  end
end

function addMaxPoolConstraint(cnn, convX, layerNum)
  m = cnn.m
  numChannels = cnn.channels[layerNum]
  convSize = cnn.layerSizes[layerNum]
  for c = 1:numChannels
    for i = 1:convSize
      for j = 1:convSize
        region = [convX[2,c,a,b] for a=2*i-1:2*i,b=2*j-1:2*j]
        @constraint(m, convX[3,c,i,j] >= region[1])
        @constraint(m, convX[3,c,i,j] >= region[2])
        @constraint(m, convX[3,c,i,j] >= region[3])
        @constraint(m, convX[3,c,i,j] >= region[4])
      end
    end
  end
end

"""
    Flatten a multidimensional layer to a vector. Corresponds to i.e. x.view(-1, 3*14*14) in ConvNN.
"""
function flattenDecConvLayer(cnn, convX, layerNum)
  sizes = cnn.layerSizes
  channels = cnn.channels
  # Dimensions to reshape (flatten) to, i.e. 14*14*3
  dims = (sizes[layerNum]*sizes[layerNum]*channels[layerNum], 1)
  layer = [convX[layerNum,c,i,j] for j=1:sizes[layerNum],i=1:sizes[layerNum],c=1:channels[layerNum]]
  return reshape(layer, dims)
end

function getRunResults(m)
  value = getobjectivevalue(m)
  solvetime = getsolvetime(m)
  bound = getobjectivebound(m)
  nodes = getnodecount(m)
  gap = abs(value - bound)/value
  return value,solvetime,bound,nodes,gap
end