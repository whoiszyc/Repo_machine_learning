using NeuralVerification
using LazySets

# old neural nets
at = @__DIR__
nnet = read_nnet("$at/networks/small_nnet.nnet")

input_set  = Hyperrectangle(low = [-1.0], high = [1.0])
output_set = Hyperrectangle(low = [-1.0], high = [70.0])
problem = Problem(nnet, input_set, output_set)

solver = BaB()
result = solve(solver, problem)
println(result)
