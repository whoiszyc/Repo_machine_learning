using NeuralVerification
using LazySets
using GLPKMathProgInterface
# using MathProgBase.SolverInterface

at = @__DIR__
small_nnet = read_nnet("$at/networks/small_nnet.nnet")

inputSet = Hyperrectangle([-1.0], [0.5])
A = Array{Float64, 2}(undef, 2,1)
A[1,1] = 1
A[2,1] = -1
outputSet = HPolytope(A[1:1, :],[18.5])
problem = Problem(small_nnet, inputSet, outputSet)
optimizer = GLPKSolverMIP()
solver = BaB(0.1, optimizer)
solve(solver, problem)
