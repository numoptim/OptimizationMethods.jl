# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Implement a simple linear least square problem.

"""
TODO - docs
"""
mutable struct GaussianLeastSquares{T, S} <: AbstractNLPModel{T, S}
    # NLPModel bare minimum
    meta :: NLPModelMeta{T, S}
    counters :: Counters

    # Data for linear system
    A :: Array{T}
    xstar :: S
    b :: S
    order :: Int64 # max(how many continuous derivatives, 2)

    # for functionality
    # A_squared :: Array{T}
    # Ab :: S
end

# constructor for a simple regression problem
function SimpleLinearLeastSquares()
    # initialize Matrix 
    A :: Array{Float64} = zeros(2, 2)
    A[1, 1], A[1, 2] = 1, 2
    A[2, 1], A[2, 2] = 2, 1

    # initialize b
    b :: Vector{Float64} = ones(2)

    # solution of this system
    xstar :: Vector{Float64} = [1 / 3, 1 / 3]

    return GaussianLeastSquares{Float64, Vector{Float64}}(NLPModelMeta(2; x0 = zeros(2)), Counters(), A, xstar, b, 2)
end

# constructor to generate a regression problem
function GaussianLeastSquares(nrow :: Int64, ncol :: Int64)

    A = randn(nrow, ncol)
    xstar = randn(ncol)
    b = A * xstar

    return GaussianLeastSquares{Float64, Vector{Float64}}(NLPModelMeta(2; x0 = zeros(2)), Counters(), A, xstar, b, 2)
end

# function to initialize data
function initialize(prog_data :: GaussianLeastSquares{T, S}) where S <: Vector{T} where T <: Real

    # compute and create storage for auxilary variables
    Asquared = prog_data.A' * prog_data.A
    Ab = prog_data.A' * prog_data.b

    function obj(prog_data :: GaussianLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
        prog_data.counters.neval_obj += 1
        return .5 * norm(prog_data.A * x - prog_data.b) ^ 2
    end

    function grad!(g :: S, prog_data :: GaussianLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
        prog_data.counters.neval_grad += 1
        mul!(g, Asquared, x)
        g .-= Ab
    end

    function hess!(H :: Array{T}, prog_data :: GaussianLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
        prog_data.counters.neval_hess += 1
        H .= @views( Asquared )
    end

    # return functions
    return obj, grad!, hess!
end

# Implementation of objective, gradient, and hessian
function obj(prog_data :: GaussianLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    prog_data.counters.neval_obj += 1
    return .5 * norm(prog_data.A * x - prog_data.b) ^ 2
end

function grad(prog_data :: GaussianLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    prog_data.counters.neval_grad += 1
    return prog_data.A' * prog_data.A * x - prog_data.A * prog_data.b
end

function grad!(g :: S, prog_data :: GaussianLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    prog_data.counters.neval_grad += 1
    mul!(g, prog_data.A' * prog_data.A, x)
    g .-= prog_data.Ab
end

function hess(prog_data :: GaussianLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    prog_data.counters.neval_hess += 1
    return prog_data.A' * prog_data.A
end

function hess!(H :: Array{T}, prog_data :: GaussianLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    prog_data.counters.neval_hess += 1
    H .= @views( prog_data.A' * prog_data.A )
end