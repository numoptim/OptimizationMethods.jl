# OptimizationMethods.jl
# Date: 2025/17/07
# Author: Christian Varner
# Purpose: Implementation of a CUTEst Wrapper file
# to gain access to it's functionality. The main
# problem addressed is creating empty Allocate, Precomp structs, and
# a initialize function

"""
"""
function CUTEstWrapper(
    ::Type{T};
    name::String
) where {T}
    nlp = CUTEstModel{Float64}(name)
    return nlp
end

"""
"""
struct PrecomputeCUTEst{T} <: AbstractPrecompute{T}
end
function PrecomputeCUTEst(progData::CUTEstModel{T}) where {T}
    return PrecomputeCUTEst{T}() 
end

"""
"""
struct AllocateCUTEst{T} <: AbstractProblemAllocate{T}
end
function AllocateCUTEst(progData::CUTEstModel{T}) where {T}
    return AllocateCUTEst{T}()
end

"""
"""
function initialize(progData::CUTEstModel{T}) where {T}
    
    precompute = PrecomputeCUTEst(progData)
    store = AllocateCUTEst(progData)

    return precompute, store
end

###############################################################################
# Operations that are not in-place. Does not make use of precomputed values.
###############################################################################

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

###############################################################################
# Operations that are in-place. Makes use of precomputed values. 
###############################################################################