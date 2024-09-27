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
    func = OptimizationMethods.GaussianLeastSquares(Float64; nequ = 10, nvar = 5)

    ####################################
    # Test First Iteration
    ####################################

    # alfa0 default value -- long true
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1)
    @test norm( xres - func.meta.x0 + (1e-4 * OptimizationMethods.grad(func, func.meta.x0)) ) < eps() * 1e2

    # alfa0 default value -- long false
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1; long = false)
    @test norm( xres - func.meta.x0 + (1e-4 * OptimizationMethods.grad(func, func.meta.x0)) ) < eps() * 1e2

    # alfa0 set -- long true
    afla0 :: Float64 = rand(1)[1]
    x0 = randn(size(func.meta.x0))
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, x0, 1; alfa0 = afla0)
    @test norm( xres - x0 + (afla0 * OptimizationMethods.grad(func, x0)) ) < eps() * 1e4
    
    # alfa0 set -- long false
    afla0 = rand(1)[1]
    x0 = randn(size(func.meta.x0))
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, x0, 1; long = false, alfa0 = afla0)
    @test norm( xres - x0 + (afla0 * OptimizationMethods.grad(func, x0)) ) < eps() * 1e5

    #####################################
    # Test if BB step size is used
    #####################################

    # long true
    x1, stats1 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1)
    x2, stats2 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 2)
    x3, stats3 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 3)

    Δx = x2 - x1
    Δg = OptimizationMethods.grad(func, x2) - OptimizationMethods.grad(func, x1)
    long_step = (Δx' * Δx) / (Δx' * Δg) 

    @test norm( x3 - x2 + (long_step * OptimizationMethods.grad(func, x2)) ) < eps() * 1e2

    # long false
    x1, stats1 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1; long = false)
    x2, stats2 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 2; long = false)
    x3, stats3 = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 3; long = false)

    Δx = x2 - x1
    Δg = OptimizationMethods.grad(func, x2) - OptimizationMethods.grad(func, x1)
    short_step = (Δx' * Δg) / (Δg' * Δg)

    @test norm( x3 - x2 + (short_step * OptimizationMethods.grad(func, x2)) ) < eps() * 1e2

    #####################################
    # Test full algorithm output
    #####################################
    
    # default tolerance - using long step size variant
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 100)
    @test norm(OptimizationMethods.grad(func, xres)) < 1e-10

    # different tolerance - using long step size variant
    tol = 1e-4 * rand(1)[1]
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 100; gradient_condition = tol)
    @test norm(OptimizationMethods.grad(func, xres)) < tol

    # default tolerance - using short step size variant
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 100; long = false)
    @test norm(OptimizationMethods.grad(func, xres)) < 1e-10

    # default tolerance - using short step size variant
    tol = 1e-4 * rand(1)[1]
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 100; gradient_condition = tol, long = false)
    @test norm(OptimizationMethods.grad(func, xres)) < tol

    #####################################
    # Check stats output
    #####################################

    # default gradient condition

    ## max_iter <= 0
    max_iter = rand(-9:0)
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, max_iter)
    @test xres == func.meta.x0
    @test stats.status_message == "max_iter is smaller than or equal to 0."
    @test stats.status[1] == -1
    @test stats.grad_norm == -1

    ## max_iter > 0 but we start with an arbitrary point
    func = OptimizationMethods.GaussianLeastSquares(Float64; nequ = 10, nvar = 5) 
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1000)
    @test stats.nobj == func.counters.neval_obj
    @test stats.ngrad == func.counters.neval_grad
    @test stats.nhess == func.counters.neval_hess
    @test abs( stats.grad_norm - norm(OptimizationMethods.grad(func, xres)) ) < eps() * 10
    @test stats.status[1] == 1
    @test stats.status[2] == 1e-10
    @test stats.status_message == "Gradient tolerance was reached."

    ## max_iter > 0 but we have a stationary point as the initial guess
    func = OptimizationMethods.GaussianLeastSquares(Float64; nequ = 10, nvar = 5) 
    xsol, statsSol = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 10000) 
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, xsol, 100) 
    @test xres ≈ xsol
    @test stats.total_iters == 0
    @test stats.nobj == statsSol.nobj
    @test stats.ngrad == statsSol.ngrad + 1
    @test stats.nhess == statsSol.nhess
    @test abs( stats.grad_norm - norm(OptimizationMethods.grad(func, xres)) ) < eps() * 10
    @test stats.nhess == 0
    @test stats.status[1] == 1
    @test stats.status[2] == 1e-10
    @test stats.status_message == "Gradient at initial point was already below tolerance."
    
    ## max_iter > 0 but we start with an arbitrary point -- make nans happen
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 10000; gradient_condition = -1.0)  
    @test stats.nobj == func.counters.neval_obj
    @test stats.ngrad == func.counters.neval_grad
    @test stats.nhess == func.counters.neval_hess
    @test abs( stats.grad_norm - norm(OptimizationMethods.grad(func, xres)) ) < eps() * 10 
    @test stats.status[1] == 0
    @test stats.status[2] == -1
    @test stats.status_message == "Termination due to step size being nan or inf."

    # random gradient condition
    gradient_condition = 1e-4 * rand(1)[1]

    ## max_iter <= 0
    max_iter = rand(-9:0)
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, max_iter; gradient_condition = gradient_condition)
    @test xres == func.meta.x0
    @test stats.status_message == "max_iter is smaller than or equal to 0."
    @test stats.status[1] == -1
    @test stats.grad_norm == -1

    ## max_iter > 0 but we start with an arbitrary point
    func = OptimizationMethods.GaussianLeastSquares(Float64; nequ = 10, nvar = 5) 
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1000; gradient_condition = gradient_condition)
    @test stats.nobj == func.counters.neval_obj
    @test stats.ngrad == func.counters.neval_grad
    @test stats.nhess == func.counters.neval_hess
    @test abs( stats.grad_norm - norm(OptimizationMethods.grad(func, xres)) ) < eps() * 10
    @test stats.status[1] == 1
    @test stats.status[2] == gradient_condition
    @test stats.status_message == "Gradient tolerance was reached."

    ## max_iter > 0 but we have a stationary point as the initial guess
    func = OptimizationMethods.GaussianLeastSquares(Float64; nequ = 10, nvar = 5) 
    xsol, statsSol = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 10000; gradient_condition = gradient_condition) 
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, xsol, 100; gradient_condition = gradient_condition) 
    @test xres ≈ xsol
    @test stats.total_iters == 0
    @test stats.nobj == statsSol.nobj
    @test stats.ngrad == statsSol.ngrad + 1
    @test stats.nhess == statsSol.nhess
    @test abs( stats.grad_norm - norm(OptimizationMethods.grad(func, xres)) ) < eps() * 10
    @test stats.nhess == 0
    @test stats.status[1] == 1
    @test stats.status[2] == gradient_condition
    @test stats.status_message == "Gradient at initial point was already below tolerance."
    
    ## max_iter > 0 but we start with an arbitrary point -- make nans happen
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 10000; gradient_condition = -1.0)  
    @test stats.nobj == func.counters.neval_obj
    @test stats.ngrad == func.counters.neval_grad
    @test stats.nhess == func.counters.neval_hess
    @test abs( stats.grad_norm - norm(OptimizationMethods.grad(func, xres)) ) < eps() * 10
    @test stats.status[1] == 0
    @test stats.status[2] == -1
    @test stats.status_message == "Termination due to step size being nan or inf."

    #####################################################################################
    ############################## Test if type is changed ##############################
    #####################################################################################
    
    # testing problem
    func = OptimizationMethods.GaussianLeastSquares(Float16)

    ####################################
    # Test First Iteration
    ####################################

    # alfa0 default value -- long true
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1)
    @test typeof(xres) == Vector{Float16}

    # alfa0 default value -- long false
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1; long = false)
    @test typeof(xres) == Vector{Float16}

    # alfa0 set -- long true
    afla01 = rand(Float16, 1)[1]
    x0 = randn(Float16, size(func.meta.x0))
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, x0, 1; alfa0 = afla01)
    @test typeof(xres) == Vector{Float16}
    
    # alfa0 set -- long false
    afla01 = rand(Float16, 1)[1]
    x0 = randn(Float16, size(func.meta.x0))
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, x0, 1; long = false, alfa0 = afla01)
    @test typeof(xres) == Vector{Float16}

    #####################################
    # Test if BB step size is used
    #####################################

    # long true
    x1, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1)
    x2, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 2)
    x3, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 3)
    
    @test typeof(x1) == Vector{Float16}
    @test typeof(x2) == Vector{Float16}
    @test typeof(x3) == Vector{Float16}

    # long false
    x1, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1; long = false)
    x2, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 2; long = false)
    x3, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 3; long = false)

    @test typeof(x1) == Vector{Float16}
    @test typeof(x2) == Vector{Float16}
    @test typeof(x3) == Vector{Float16} 

    #####################################
    # Test full algorithm output
    #####################################
    
    # using long step size variant
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1000)
    @test typeof(xres) == Vector{Float16}

    # using short step size variant
    xres, stats = OptimizationMethods.barzilai_borwein_gd(func, func.meta.x0, 1000; long = false)
    @test typeof(xres) == Vector{Float16}
end

end