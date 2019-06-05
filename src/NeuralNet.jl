#=
NeuralNet:
- Julia version: 1.1.0
- Author: veerle
- Date: 2019-06-03
=#

abstract type NeuralNet end
struct NN <: NeuralNet
    m
    input
    w
    b
    layerSizes
    targetLabel
end

struct CNN <: NeuralNet
    m
    input
    w
    b
    layerSizes
    targetLabel
    fc
    conv
end


 """
    Build the MILP model for a (D)NN by adding constraints for the neural network, then optimize.
"""
function (nn::NN)(doAdversarial::Bool)
  m = nn.m
  numLayers = size(nn.layerSizes,1)

  # Initialize x decision variable, x >= 0. x[k,j] is x^k_j. Arbitrary upper bound of 10000
  @variable(m, 0 <= x[k = 1:numLayers, j = 1:nn.layerSizes[k]] <= 10000)
  # Initialize s decision variable, s >= 0. s[k,j] is s^k_j
  @variable(m, 0 <= s[k = 2:numLayers, j = 1:nn.layerSizes[k]] <= 10000)

  if doAdversarial
    d = addAdversarialConstraints(nn, x)

    # Constrain output to be equal to expected output
    output = [x[numLayers, j] for j in 1:nn.layerSizes[numLayers, ]]
    label = nn.targetLabel
    for k in 1:nn.layerSizes[numLayers]
      if k != label+1
        @constraint(m, x[numLayers,label+1] >= 1.2*x[numLayers,k])
      end
    end
    # Minimize distance between original input and adversarial input
    @objective(m, Min, sum(d))
  else
    # First x layer must be equal to the input
    @constraint(m, [x[1,j] for j in 1:nn.layerSizes[1]] .== nn.input)

    # Minimize over all x. Since input is fixed, this does not really do anything
    @objective(m, Min, sum(x))
  end

  add_relu_constraints(nn, numLayers, x, s)

  print("Solver time: $(@elapsed(solve(m)))")
  return m,x,s
end

function(cnn::CNN)(doAdversarial::Bool)
    print("IMPLEMENT ME!")
end
