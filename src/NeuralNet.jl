#=
NeuralNet:
- Julia version: 1.1.0
- Author: veerle
- Date: 2019-06-03
=#

using JuMP, ConditionalJuMP


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
    convW
    w
    convB
    b
    layerSizes
    targetLabel
    channels
    numConv
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

  firstLayer = [x[1,j] for j in 1:nn.layerSizes[1]]

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
    @constraint(m, firstLayer .== nn.input)

    # Minimize over all x. Since input is fixed, this does not really do anything
    @objective(m, Min, sum(x))
  end

  add_relu_constraints(nn, 2, numLayers, x, s, firstLayer)

  print("Solver time: $(@elapsed(solve(m)))")
  return m,x,s
end

function(cnn::CNN)(doAdversarial::Bool)
    m = cnn.m
    numLayers = size(cnn.layerSizes,1)
    sizes = cnn.layerSizes
    channels = cnn.channels
    numConv = cnn.numConv
    println(sizes) # [28+2,28,14,100,10]

    # Add conv decision variables, input included. Both x and s, as relu is used before pooling
    @variable(m, 0 <= convX[k=1:numConv,c=1:channels[k],i = 1:sizes[k],j = 1:sizes[k]] <= 10000)
    @variable(m, 0 <= convS[k=2:numConv,c=1:channels[k],i = 1:sizes[k],j = 1:sizes[k]] <= 10000)

    # Add fully connected decision varaibles
    @variable(m, 0 <= fcX[k = numConv+1:numLayers, j = 1:sizes[k]] <= 10000)
    @variable(m, 0 <= fcS[k = numConv+1:numLayers, j = 1:sizes[k]] <= 10000)

    # Constraint padding to be 0
    addPaddingConstraint(cnn, convX, sizes[1])

    # Constrain input to be equal to input. Note that padding is not part of this.
    @constraint(m, [convX[1,1,i,j] for i=2:sizes[1]-1,j=2:sizes[1]-1] .== cnn.input)

    # First convolution!
    for c = 1:channels[2]
      w = cnn.convW["1"][string(c)]
      b = cnn.convB["1"][c]
      for i = 1:sizes[2]
        for j = 1:sizes[2]
          region = [convX[1,1,m,n] for m=i:i+2,n=j:j+2]
          # Relu constraint for convolution calculation
          @constraint(m, convX[2,c,i,j] - convS[2,c,i,j] == sum(w .* region) + b)
        end
      end
    end

    # Add average pooling constraint
    addAveragePoolConstraint(cnn, convX, 3)

    # Flatten the layer
    layer3X = flattenDecConvLayer(cnn, convX, 3)

    # Add the ReLU constraints for the final fully connected layers
    add_relu_constraints(cnn, numConv+1, numLayers, fcX, fcS, layer3X)

    @objective(m, Min, sum(convX) + sum(fcX))
    solve(m)
    return m,convX,fcX,convS,fcS
end
