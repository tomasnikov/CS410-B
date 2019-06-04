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
function add_relu_constraints(nn::NeuralNet, numLayers, x, s)
  for k in 2:numLayers
    # x_j and s_j are values for layer k, j in 1:n_k
    layerX = [x[k,j] for j in 1:nn.layerSizes[k]]
    layerS = [s[k,j] for j in 1:nn.layerSizes[k]]

    weights = nn.w[string(k-1)]
    biases = nn.b[string(k-1)]
    lastLayer = [x[k-1,j] for j in 1:nn.layerSizes[k-1]]

    # ReLU formulation, w^{k-1}^T * x^{k-1} + b^{k-1} = x^k - s^k
    @constraint(nn.m, transpose(weights) * lastLayer + biases .== layerX .- layerS)

    # Either s or x must be 0
    for j in 1:nn.layerSizes[k]
      @disjunction(nn.m, (x[k,j] == 0), (s[k,j] == 0))
    end
  end
end

"""
    Introduce d variable, represents distance between input and adversarial input
"""
function addAdversarialConstraints(nn, x)
    @variable(nn.m, 0 <= d[1,j=1:nn.layerSizes[1]] <= 10000)

    firstLayerX = [x[1,j] for j in 1:nn.layerSizes[1]]
    firstLayerD = [d[1,j] for j in 1:nn.layerSizes[1]]

    # Input must be in range 0 to 1
    @constraint(nn.m, firstLayerX .<= .1)
    # Constrain d to be distance between original input and x input
    @constraint(nn.m, -firstLayerD .<= firstLayerX - nn.input)
    @constraint(nn.m, firstLayerD .>= firstLayerX - nn.input)
    return d
end