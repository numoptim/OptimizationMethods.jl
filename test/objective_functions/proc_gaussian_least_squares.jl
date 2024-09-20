# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Test the functionality of simple least squares
# implemented in src/objective_functions/simple_least_squares.jl

module ProceduralSimpleLeastSquares

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Simple Least Squares Example -- Procedural" begin

    # set testing context
    Random.seed!(1010)

    # start with default constructor
    func = OptimizationMethods.SimpleLinearLeastSquares()
    
    A :: Array{Float64} = zeros(2, 2)
    A[1, 1], A[1, 2] = 1, 2
    A[2, 1], A[2, 2] = 2, 1
    b = ones(2)

    # test contents and definitions
    @test func.A == A
    @test func.b == ones(2)
    @test func.xstar == [1 / 3, 1 / 3]
    @test norm( func.A * func.xstar - func.b ) < eps()
    @test norm( A' * A - func.A_squared ) < eps()
    @test norm( A' * b - func.Ab ) < eps()
    @test func.meta.x0 == zeros(2)

    # test functionality
    x = randn(2) 
    @test abs(obj(func, x) - (.5 * norm(A * x - b) ^ 2)) < eps()
    @test norm(grad(func, x) - (func.A_squared * x - func.Ab) ) < eps()
    @test norm(hess(func, x) - func.A_squared) < eps()

    gk = zeros(2)
    grad!(gk, func, x)
    @test norm(gk - (func.A_squared * x - func.Ab) ) < eps()
    
    H = zeros(2, 2)
    hess!(H, func, x)
    @test norm(H - func.A_squared) < eps()

    @test func.counters.neval_obj == 1
    @test func.counters.neval_grad == 2
    @test func.counters.neval_hess == 2

    # non-default constructor
    dim = 10
    A = randn(10, 10)
    xstar = randn(10)
    b = A * xstar

    # constructor
    meta = OptimizationMethods.NLPModelMeta{Float64, Vector{Float64}}(dim; x0 = zeros(dim))
    func = OptimizationMethods.SimpleLinearLeastSquares{Float64, Vector{Float64}}(
        meta,
        OptimizationMethods.Counters(),
        A,
        xstar,
        b,
        A' * A,
        A' * b
    )

    # test contents and definitions
    @test func.A == A
    @test func.b == b
    @test func.xstar == xstar
    @test norm( func.A * func.xstar - func.b ) < eps()
    @test norm( A' * A - func.A_squared ) < eps()
    @test norm( A' * b - func.Ab ) < eps()
    @test func.meta.x0 == zeros(dim)

    # test functionality
    x = randn(dim) 
    @test abs(obj(func, x) - (.5 * norm(A * x - b) ^ 2)) < eps()
    @test norm(grad(func, x) - (func.A_squared * x - func.Ab) ) < eps()
    @test norm(hess(func, x) - func.A_squared) < eps()

    gk = zeros(dim)
    grad!(gk, func, x)
    @test norm(gk - (func.A_squared * x - func.Ab) ) < eps()

    H = zeros(dim, dim)
    hess!(H, func, x)
    @test norm(H - func.A_squared) < eps()

    @test func.counters.neval_obj == 1
    @test func.counters.neval_grad == 2
    @test func.counters.neval_hess == 2 
end

end