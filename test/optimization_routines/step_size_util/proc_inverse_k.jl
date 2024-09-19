# Date: 09/19/2024
# Author: Christian Varner
# Purpose: Implement tests for the inverse k step size
# function in src/optimization_routines/step_size_util

module ProceduralInverseK

using Test, OptimizationMethods, Random

@testset "Inverse K -- Procedural" begin

    # testing context
    Random.seed!(1010)
    
    # Initialize the step size function
    c = abs(randn(1)[1])
    step_size = OptimizationMethods.inverse_k(c)

    # test first iteration
    @test c == step_size(1)

    # test out a random iteration number
    iter = rand(2:10000)
    @test c / iter ≈ step_size(iter)
    @test c / (iter + 1) ≈ step_size(iter + 1)

end

end # End Module