# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Implement a simple linear least square problem.

"""
TODO - docs
"""
mutable struct SimpleLinearLeastSquares{T, S} <: AbstractNLPModel{T, S}
    # NLPModel bare minimum
    meta :: NLPModelMeta{T, S}
    counters :: Counters

    # Data for linear system
    A :: Array{T}
    xstar :: S
    b :: S

    # for functionality
    A_squared :: Array{T}
    Ab :: S
end

function SimpleLinearLeastSquares()
    # initialize Matrix 
    A :: Array{Float64} = zeros(2, 2)
    A[1, 1], A[1, 2] = 1, 2
    A[2, 1], A[2, 2] = 2, 1

    # initialize b
    b :: Vector{Float64} = ones(2)

    # solution of this system
    xstar :: Vector{Float64} = [1 / 3, 1 / 3]

    return SimpleLinearLeastSquares{Float64, Vector{Float64}}(NLPModelMeta(2; x0 = zeros(2)), Counters(), A, xstar, b, A'*A, A'*b)
end

# TODO: Implement a better constructor so that don't have to specify meta etc.

# Implementation of objective, gradient, and hessian
function obj(func :: SimpleLinearLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    func.counters.neval_obj += 1
    return .5 * norm(func.A * x - func.b) ^ 2
end

function grad(func :: SimpleLinearLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    func.counters.neval_grad += 1
    return func.A_squared * x - func.Ab
end

function grad!(g :: S, func :: SimpleLinearLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    func.counters.neval_grad += 1
    mul!(g, func.A_squared, x)
    g .-= func.Ab
end

function hess(func :: SimpleLinearLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    func.counters.neval_hess += 1
    return func.A_squared
end

function hess!(H :: Array{T}, func :: SimpleLinearLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    func.counters.neval_hess += 1
    H .= @views( func.A_squared )
end