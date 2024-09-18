# Date: 09/18/2024
# Author: Christian Varner
# Test the implementation of diminishing step size gd
# implemented in src/optimization_routines/diminishing_step_size_gd.jl

module ProceduralDiminishingStepSizeGD

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Diminishing Step Size GD -- Procedural" begin

    # testing context
    Random.seed!(1010)

    # testing problem
    func = OptimizationMethods.SimpleLinearLeastSquares()

end

end