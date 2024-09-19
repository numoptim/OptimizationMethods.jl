# Date: 09/19/2024
# Author: Christian Varner
# Purpose: Implement tests for the root k step size
# function in src/optimization_routines/step_size_util

module ProceduralRootK

using Test, OptimizationMethods, Random

@testset "Root K -- Procedural" begin

    # testing context
    Random.seed!(1010)

    # Initialize the step size function
    c = abs(rand(1)[1])
    step_size = OptimizationMethods.root_k(c)

    # test first iteration
    @test c == step_size(1)

    # test out a random iteration number
    iter = rand(2:10000)
    @test c / sqrt(iter) ≈ step_size(iter)
    @test c / sqrt(iter + 1) ≈ step_size(iter + 1)

end

end # end module