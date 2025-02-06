# Date: 02/05/2025
# Author: Christian Varner
# Purpose: Implementation of gradient descent using a basic
# version of Wolfe line search

"""
    WolfeGD{T} <: AbstractOptimizerData{T}

Mutable `struct` that represents gradient descent using the ELBS routine to
satisfy weak wolfe conditions. 

# Fields

# Constructors

## Arguments

## Keyword Arguments
"""
mutable struct WolfeEBLSGD{T} <: AbstractOptimizerData{T}
    name::String
    α::T
    δ::T
    c1::T
    c2::T
    line_search_max_iterations::Int
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function WolfeEBLSGD(
    ::Type{T};
    x0::Vector{T},
    α::T,
    δ::T,
    c1::T,
    c2::T, 
    line_search_max_iterations::Int,
    threshold::T,
    max_iterations::Int
) where {T}

    # name of the optimizer
    name = "Gradient Descent with Line Search for Weak Wolfe Conditions"

    # initialize iter_hist and grad_val_hist
    d::Int64 = length(x0)
    iter_hist::Vector{Vector{T}} = 
        Vector{Vector{T}}([Vector{T}(undef, d) for i in 1:(max_iterations + 1)])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1) 
    stop_iteration::Int64 = -1 ## dummy value

    return WolfeEBLSGD(name, α, δ, c1, c2, line_search_max_iterations, threshold,
        max_iterations, iter_hist, grad_val_hist, stop_iteration)
end

"""
    TODO
"""
function wolfe_gd(
    optData::WolfeEBLSGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
)
end