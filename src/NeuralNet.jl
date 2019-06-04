#=
NeuralNet:
- Julia version: 1.1.0
- Author: veerle
- Date: 2019-06-03
=#

struct NeuralNet
    m
    input
    w
    b
    layerSizes
    targetLabel
end