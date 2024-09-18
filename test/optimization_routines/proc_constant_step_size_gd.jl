# Date 09/18/2024
# Author: Christian Varner
# Purpose: Test the constant step size gradient descent
# algorithm implemented in src/optimization_routines/constant_step_size_gd.jl

module ProceduralConstantStepSizeGD

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Constant Step Size GD -- Procedural" begin

    # set the seed
    Random.seed!(1010)

    # testing problem
    func = OptimizationMethods.SimpleLinearLeastSquares()

    # test that default step size is 1e-4
    xres = constant_step_size_gd(func, func.meta.x0, 1)
    Δx = xres - func.meta.x0
    @test norm(Δx - (-1e-4 * grad(func, func.meta.x0)) ) < eps()

    # set the initial set size
    alfa0 = 1/9 # 1/L

    # test first iteration
    xres = constant_step_size_gd(func, func.meta.x0, 1; alfa0 = alfa0)
    Δx = xres - func.meta.x0
    @test norm(Δx - (-alfa0 * grad(func, func.meta.x0)) ) < eps()

    # test random iteration number -- step size default
    iteration = rand(2:10)
    xiter_prev = constant_step_size_gd(func, func.meta.x0, iteration - 1)
    xiter = constant_step_size_gd(func, func.meta.x0, iteration)
    Δx = xiter - xiter_prev
    @test norm(Δx - (-1e-4 * grad(func, xiter_prev))) < eps()

    # test random iteration number -- step size set
    iteration = rand(2:10)
    xiter_prev = constant_step_size_gd(func, func.meta.x0, iteration - 1; alfa0 = alfa0)
    xiter = constant_step_size_gd(func, func.meta.x0, iteration; alfa0 = alfa0)
    Δx = xiter - xiter_prev
    @test norm(Δx - (-alfa0 * grad(func, xiter_prev))) < eps()
end

end # end module