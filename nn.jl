using JuMP
using Gurobi
using Distributions
using LinearAlgebra
using DelimitedFiles

function constraints_and_optimize(initX, z, w, b, c, g, expOutput)
  
  m = Model(with_optimizer(Gurobi.Optimizer))

  @variable(m, x[1:2,1:1] >= 0)
  @variable(m, s[1:2,1:1] >= 0)

  @constraint(m, x .== initX)

  #@variable(m, output1[1:2,1:1])
  @variable(m, layer1[1:size(w[1],2), 1:size(x,2)])
  @variable(m, layer2[1:size(w[2],2), 1:size(layer1,2)])

  @constraint(m, layer1 - s .== w[1] * x + b[1])
  @constraint(m, layer2 .== transpose(w[2]) * layer1 + b[2])

  @constraint(m, layer2 .== expOutput)

  @objective(m, Min, sum(x .* c) + sum(z .* g))

  optimize!(m)
  return m,x,s,layer2
end

function print_vars(m,x,s,z,w,b,c,g,output)
  
  println(" ")
  println("Objective Value: ", JuMP.objective_value(m))
  println("X Values: ")
  for i in 1:size(x,1)
    println(JuMP.value(x[i]))
  end

  println("S Values: ")
  for i in 1:size(s,1)
    println(JuMP.value(s[i]))
  end

  println("Weight Values: ")
  for l in 1:size(w,1)
    println("Layer ", l, ':')
    for i in 1:size(w[l],1)
      for j in 1:size(w[l],2)
        print(w[l][i,j], " ")
      end
      println(" ")
    end
  end

  println("Bias Values: ")
  for l in 1:size(b,1)
    println("Layer ", l, ':')
    for i in 1:size(b[l],1)
      println(b[l][i])
    end
  end

  println(size(w), size(x))
  println(size(output))
  println("Output: ", JuMP.value(output[1]))

end

function main()
  #file = ARGS[1]

  #data = readdlm(file, ',', Int, '\n')
  x = [0.0; 1.0]
  expOutput = [0.0]
  z = [0; 0]
  w1 = [1.0 1.0;1.0 1.0]
  w2 = [1.0; -2.0]
  b1 = [0.0; -1.0]
  b2 = [0.0]
  w = [w1, w2]
  b = [b1, b2]
  c = [1; 1]
  g = [0; 0]


  t = @elapsed m,x,s,output = constraints_and_optimize(x,z,w,b,c,g,expOutput)
  println(" ")
  println("Time: ",t)
  print_vars(m,x,s,z,w,b,c,g,output)
end


main()