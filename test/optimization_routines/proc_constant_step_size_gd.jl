# Date 09/18/2024
# Author: Christian Varner
# Purpose: Test the constant step size gradient descent
# algorithm implemented in src/optimization_routines/OptimizationMethods.constant_step_size_gd.jl

module ProceduralConstantStepSizeGD

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Constant Step Size GD -- Procedural" begin

    # set the seed
    Random.seed!(1010)

    # testing problem
    func = OptimizationMethods.GaussianLeastSquares(Float64; nequ = 100, nvar = 50)

    ##############################################################
    # Test first iteration of method
    ##############################################################

    # Default value
    xres = OptimizationMethods.constant_step_size_gd(func, func.meta.x0, 1)
    Δx = xres - func.meta.x0
    @test norm(Δx - (-1e-4 * OptimizationMethods.grad(func, func.meta.x0)) ) < eps() * 1e3

    # Custom value
    alfa0 = rand(1)[1]
    xres = OptimizationMethods.constant_step_size_gd(func, func.meta.x0, 1; alfa0 = alfa0)
    Δx = xres - func.meta.x0
    @test norm(Δx - (-alfa0 * OptimizationMethods.grad(func, func.meta.x0)) ) < eps() * 1e3

    ##############################################################
    # Test random iteration
    ##############################################################
    
    # step size default
    iteration = rand(2:10)
    xiter_prev = OptimizationMethods.constant_step_size_gd(func, func.meta.x0, iteration - 1)
    xiter = OptimizationMethods.constant_step_size_gd(func, func.meta.x0, iteration)
    Δx = xiter - xiter_prev
    @test norm(Δx - (-1e-4 * OptimizationMethods.grad(func, xiter_prev))) < eps() * 1e3

    # step size set
    iteration = rand(2:10)
    alfa0 = 1e-4 * rand(1)[1]
    xiter_prev = OptimizationMethods.constant_step_size_gd(func, func.meta.x0, iteration - 1; alfa0 = alfa0)
    xiter = OptimizationMethods.constant_step_size_gd(func, func.meta.x0, iteration; alfa0 = alfa0)
    Δx = xiter - xiter_prev
    d = - alfa0 * OptimizationMethods.grad(func, xiter_prev) 
    @test norm( xiter - (xiter_prev + d) ) < eps() * 1e3

    ##############################################################
    # Test random iteration
    ##############################################################

    # test full output
    func = OptimizationMethods.GaussianLeastSquares(Float64; nequ = 10, nvar = 5)
    A = func.coef
    alfa0 = 1 / maximum(eigen(A' * A).values)

    xres = OptimizationMethods.constant_step_size_gd(func, func.meta.x0, 1000; alfa0 = alfa0)
    g_xres = norm( OptimizationMethods.grad(func, xres) )
    @test g_xres < eps() * 1e3
end

end # end module