# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Test barzilai-borwein with gradient descent
# implemented in src/optimization_routines/barzilai-borwein-gd.jl

module ProceduralBarzilaiBorweinGD

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Barzilai Borwein GD -- Procedural" begin

    # testing context
    Random.seed!(1010)

    # testing problem
    func = OptimizationMethods.SimpleLinearLeastSquares()

    # test first iteration -- default value
    xres = barzilai_borwein_gd(func, func.meta.x0, 1)
    @test norm( xres - func.meta.x0 + (1e-4 * grad(func, func.meta.x0)) ) < eps()

    # test first iteration -- other value
    afla0 :: Float64 = rand(1)[1]
    xres = barzilai_borwein_gd(func, func.meta.x0, 1; alfa0 = afla0)
    @test norm( xres - func.meta.x0 + (afla0 * grad(func, func.meta.x0)) ) < eps() 

    # test first three iterations -- default value
    x1 = barzilai_borwein_gd(func, func.meta.x0, 1)
    x2 = barzilai_borwein_gd(func, func.meta.x0, 2)
    x3 = barzilai_borwein_gd(func, func.meta.x0, 3)

    Δx = x2 - x1
    Δg = grad(func, x2) - grad(func, x1)
    long_step = (Δx' * Δx) / (Δx' * Δg) 

    @test norm( x3 - x2 + (long_step * grad(func, x2)) ) < eps()

    # test first three iterations -- short step
    x1 = barzilai_borwein_gd(func, func.meta.x0, 1; long = false)
    x2 = barzilai_borwein_gd(func, func.meta.x0, 2; long = false)
    x3 = barzilai_borwein_gd(func, func.meta.x0, 3; long = false)

    Δx = x2 - x1
    Δg = grad(func, x2) - grad(func, x1)
    short_step = (Δx' * Δg) / (Δg' * Δg)

    @test norm( x3 - x2 + (long_step * grad(func, x2)) ) < eps()

    # test full algorithm
    xres = barzilai_borwein_gd(func, func.meta.x0, 100)
    @test norm(xres - func.xstar) < eps()

    xres = barzilai_borwein_gd(func, func.meta.x0, 100; long = false)
    @test norm(xres - func.xstar) < eps() 
end

end