using Gurobi, JuMP, CPLEX
import MathProgBase
const MPB = MathProgBase


"loadLP loads a model from file, initializing the model using the (optional) solver"
function loadLP(filename, solver=CplexSolver())
    model =MathProgBase.LinearQuadraticModel(solver)
    MPB.loadproblem!(model, "fichetti_data/dnn1_1sec/dnn1_1sec_0.lp")

    return model
end

lp = loadLP("fichetti_data/dnn1_1sec/dnn1_1sec_0.lp")
MPB.optimize!(lp)