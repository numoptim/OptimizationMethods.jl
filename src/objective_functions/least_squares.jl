# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Implement a simple linear least square problem.

"""
TODO - docs
"""
mutable struct LinearLeastSquares{T, S} <: AbstractNLPModel{T, S}
    # NLPModel bare minimum
    meta :: NLPModelMeta{T, S}
    counters :: Counters

    # Data for linear system
    A :: Array{T}
    b :: S
end

function LinearLeastSquares()
    # initialize Matrix 
    A :: Array{Float64} = zeros(2, 2)
    A[1, 1], A[1, 2] = 1, 2
    A[2, 1], A[2, 2] = 2, 1

    # initialize b
    b :: Vector{Float64} = ones(2)

    return LinearLeastSquares{Float64, Vector{Float64}}(NLPModelMeta(2; x0 = zeros(2)), Counters(), A, b)
end

# Implementation of objective, gradient, and hessian
function obj(func :: LinearLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    return .5 * norm(func.A * x - func.b) ^ 2
end

function grad(func :: LinearLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    return func.A' * func.A * x - func.A' * func.b
end

function hess(func :: LinearLeastSquares{T, S}, x :: S) where S <: Vector{T} where T <: Real
    return func.A' * func.A
end