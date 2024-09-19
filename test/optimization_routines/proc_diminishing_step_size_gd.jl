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

    #####################################
    # Test with default step size func
    #####################################
    
    # test with default function -- one iteration
    x0 = func.meta.x0
    x1 = diminishing_step_size_gd(func, func.meta.x0, 1)
    @test norm( x1 - x0 + grad(func, x0) ) < eps()

    # test random iteration -- default step size function
    iter = rand(2:10)
    xprev = diminishing_step_size_gd(func, func.meta.x0, iter - 1)
    xiter = diminishing_step_size_gd(func, func.meta.x0, iter)
    @test norm(xiter - xprev + (1 / iter) * grad(func, xprev)) < eps()

    # test convergence -- default step size function
    xres = diminishing_step_size_gd(func, func.meta.x0, 1000)
    @test norm(xres - func.xstar) < eps()

    #####################################
    # Test with different step size func
    #####################################

    # test with different step size function -- one iteration
    step_size = OptimizationMethods.root_k(2) # 2/sqrt(k)
    x1 = diminishing_step_size_gd(func, func.meta.x0, 1; step_size = step_size)
    @test norm(x1 - x0 + (step_size(1)) * grad(func, x0)) < eps()

    # test random iteration -- different step size function
    iter = rand(5:10)
    xprev = diminishing_step_size_gd(func, func.meta.x0, iter - 1; step_size = step_size)
    xiter = diminishing_step_size_gd(func, func.meta.x0, iter; step_size = step_size)
    @test norm(xiter - xprev + (step_size(iter)) * grad(func, xprev)) < eps()

    # test convergence -- different step size function
    xres = diminishing_step_size_gd(func, func.meta.x0, 1000; step_size = step_size)
    @test norm(xres - func.xstar) < eps()
end

end