# Date: 02/05/2025
# Author: Christian Varner
# Purpose: Implementation of gradient descent using a basic
# version of Wolfe line search

"""
    TODO
"""
mutable struct WolfeGD{T} <: AbstractOptimizerData{T}
end
function WolfeGD(
    ::Type{T};
    x0::Vector{T},
    # TODO - algorithm specific parameters
    threshold::T,
    max_iterations::Int
)
end

"""
    TODO
"""
function wolfe_gd(
    optData::WolfeGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
)
end