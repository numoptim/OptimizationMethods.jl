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
    func = OptimizationMethods.GaussianLeastSquares(Float64)

    ####################################
    # Test First Iteration
    ####################################

    # alfa0 default value -- long true
    xres = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1)
    @test norm( xres - func.meta.x0 + (1e-4 * OptimizationMethods.grad(func, func.meta.x0)) ) < eps() * 1e2

    # alfa0 default value -- long false
    xres = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1; long = false)
    @test norm( xres - func.meta.x0 + (1e-4 * OptimizationMethods.grad(func, func.meta.x0)) ) < eps() * 1e2

    # alfa0 set -- long true
    afla0 :: Float64 = rand(1)[1]
    x0 = randn(size(func.meta.x0))
    xres = OptimizationMethods.barzilai_borwein_gd(func, x0, 1; alfa0 = afla0)
    @test norm( xres - x0 + (afla0 * OptimizationMethods.grad(func, x0)) ) < eps() * 1e4
    
    # alfa0 set -- long false
    afla0 = rand(1)[1]
    x0 = randn(size(func.meta.x0))
    xres = OptimizationMethods.barzilai_borwein_gd(func, x0, 1; long = false, alfa0 = afla0)
    @test norm( xres - x0 + (afla0 * OptimizationMethods.grad(func, x0)) ) < eps() * 1e5

    #####################################
    # Test if BB step size is used
    #####################################

    # long true
    x1 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1)
    x2 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 2)
    x3 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 3)

    Δx = x2 - x1
    Δg = OptimizationMethods.grad(func, x2) - OptimizationMethods.grad(func, x1)
    long_step = (Δx' * Δx) / (Δx' * Δg) 

    @test norm( x3 - x2 + (long_step * OptimizationMethods.grad(func, x2)) ) < eps() * 1e2

    # long false
    x1 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1; long = false)
    x2 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 2; long = false)
    x3 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 3; long = false)

    Δx = x2 - x1
    Δg = OptimizationMethods.grad(func, x2) - OptimizationMethods.grad(func, x1)
    short_step = (Δx' * Δg) / (Δg' * Δg)

    @test norm( x3 - x2 + (short_step * OptimizationMethods.grad(func, x2)) ) < eps() * 1e2

    #####################################
    # Test full algorithm output
    #####################################
    
    # using long step size variant
    xres = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 100)
    @test norm(OptimizationMethods.grad(func, xres)) < eps() * 1e4 

    # using short step size variant
    xres = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 100; long = false)
    @test norm(OptimizationMethods.grad(func, xres)) < eps() * 1e4
    
    #####################################################################################
    ############################## Test if type is changed ##############################
    #####################################################################################
    
    # testing problem
    func = OptimizationMethods.GaussianLeastSquares(Float16)

    ####################################
    # Test First Iteration
    ####################################

    # alfa0 default value -- long true
    xres = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1)
    @test typeof(xres) == Vector{Float16}

    # alfa0 default value -- long false
    xres = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1; long = false)
    @test typeof(xres) == Vector{Float16}

    # alfa0 set -- long true
    afla01 = rand(Float16, 1)[1]
    x0 = randn(Float16, size(func.meta.x0))
    xres = OptimizationMethods.barzilai_borwein_gd(func, x0, 1; alfa0 = afla01)
    @test typeof(xres) == Vector{Float16}
    
    # alfa0 set -- long false
    afla01 = rand(Float16, 1)[1]
    x0 = randn(Float16, size(func.meta.x0))
    xres = OptimizationMethods.barzilai_borwein_gd(func, x0, 1; long = false, alfa0 = afla01)
    @test typeof(xres) == Vector{Float16}

    #####################################
    # Test if BB step size is used
    #####################################

    # long true
    x1 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1)
    x2 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 2)
    x3 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 3)
    
    @test typeof(x1) == Vector{Float16}
    @test typeof(x2) == Vector{Float16}
    @test typeof(x3) == Vector{Float16}

    # long false
    x1 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1; long = false)
    x2 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 2; long = false)
    x3 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 3; long = false)

    @test typeof(x1) == Vector{Float16}
    @test typeof(x2) == Vector{Float16}
    @test typeof(x3) == Vector{Float16} 

    #####################################
    # Test full algorithm output
    #####################################
    
    # using long step size variant
    xres = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1000)
    @test typeof(xres) == Vector{Float16}

    # using short step size variant
    xres = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1000; long = false)
    @test typeof(xres) == Vector{Float16}
end

end