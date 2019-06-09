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
    fcW
    convB
    fcB
    layerSizes
    targetLabel
    channels
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
    println("IMPLEMENT ME!")
    m = cnn.m
    numLayers = size(cnn.layerSizes,1)
    channels = cnn.channels
    convSizes = cnn.layerSizes ./ channels
    convSizes = [Int(sqrt(c)) for c in convSizes[1:3]]
    convSizes[1] += 2
    sizes = [convSizes; cnn.layerSizes[4:5]]

    # FIX PADDING IN OTHER LAYERS
    @variable(m, 0 <= convX[k=1:3,c=1:channels[k],i = 1:sizes[k],j = 1:sizes[k]] <= 10000)
    @variable(m, 0 <= convS[k=2:3,c=1:channels[k],i = 1:sizes[k],j = 1:sizes[k]] <= 10000)

    @variable(m, 0 <= fcX[k=4:numLayers,j = 1:sizes[k]] <= 10000)
    @variable(m, 0 <= fcS[k = 4:numLayers, j = 1:cnn.layerSizes[k]] <= 10000)

    @constraint(m, [convX[1,1,1,j] for j=1:30] .== 0)
    @constraint(m, [convX[1,1,30,j] for j=1:30] .== 0)
    @constraint(m, [convX[1,1,i,1] for i=1:30] .== 0)
    @constraint(m, [convX[1,1,i,30] for i=1:30] .== 0)
    @constraint(m, [convX[1,1,i,j] for i=2:29,j=2:29] .== transpose(reshape(cnn.input, (28,28))))

    convW1 = cnn.convW["1"]
    convB1 = cnn.convB["1"]

    for c = 1:channels[2]
      w = convW1[string(c)]
      b = convB1[c]
      for i = 1:convSizes[2]
        for j = 1:convSizes[2]
          region = [convX[1,1,m,n] for m=i:i+2,n=j:j+2]
          @constraint(m, convX[2,c,i,j] - convS[2,c,i,j] == sum(w .* region) + b)
        end
      end
    end

    for c = 1:channels[3]
      for i = 1:convSizes[3]
        for j = 1:convSizes[3]
          region = [convX[2,c,m,n] for m=2*i-1:2*i,n=2*j-1:2*j]
          # Following is for MaxPool
          #@constraint(m, convX[3,c,i,j] >= region[1])
          #@constraint(m, convX[3,c,i,j] >= region[2])
          #@constraint(m, convX[3,c,i,j] >= region[3])
          #@constraint(m, convX[3,c,i,j] >= region[4])
          # Average Pool instead:
          @constraint(m, convX[3,c,i,j] == sum(region)/4)
        end
      end
    end

    # This is backwards, i.e. j,i,c for some reason. Corresponds to x.view(-1, 3*14*14) in ConvNN.
    layer3X = reshape([convX[3,c,i,j] for j=1:sizes[3],i=1:sizes[3],c=1:channels[3]], (588,1))
    
    layer4X = [fcX[4,j] for j=1:sizes[4]]
    layer4S = [fcS[4,j] for j=1:sizes[4]]

    w4 = cnn.fcW["1"]
    b4 = cnn.fcB["1"]

    @constraint(m, transpose(w4) * layer3X + b4 .== layer4X .- layer4S)

    for j in 1:sizes[4]
      @disjunction(m, (fcX[4,j] == 0), (fcS[4,j] == 0))
    end

    layer5X = [fcX[5,j] for j=1:sizes[5]]
    layer5S = [fcS[5,j] for j=1:sizes[5]]

    w5 = cnn.fcW["2"]
    b5 = cnn.fcB["2"]

    @constraint(m, transpose(w5) * layer4X + b5 .== layer5X .- layer5S)

    for j in 1:sizes[5]
      @disjunction(m, (fcX[5,j] == 0), (fcS[5,j] == 0))
    end

    @objective(m, Min, sum(convX) + sum(fcX))
    solve(m)
    return m,convX, fcX,fcS
end
