# OptimizationMethods.jl
# Date: 2025/17/07
# Author: Christian Varner
# Purpose: Implementation of a CUTEst Wrapper file
# to gain access to it's functionality. The main
# problem addressed is creating empty Allocate, Precomp structs, and
# a initialize function

"""
"""
struct PrecomputeCUTEst{T} <: AbstractPrecompute{T}
end
function PrecomputeCUTEst(progData::CUTEstModel{T}) where T
    return PrecomputeCUTEst{T}() 
end

"""
"""
struct AllocateCUTEst{T} <: AbstractProblemAllocate{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocateCUTEst(progData::CUTEstModel{T}) where T
    nvar = progData.meta.nvar
    return AllocateCUTEst{T}(
        zeros(T, nvar),
        zeros(T, nvar, nvar)
    )
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

args = [
    :(progData::CUTEstModel{T}),
    :(x::Vector{T})
]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at the value `x`.
    """
    function obj($(args...)) where {T}
        return NLPModels.obj(progData, x)
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`.
    """
    function grad($(args...)) where {T}
        return NLPModels.grad(progData, x)
    end

    @doc """
         objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the objective function and gradient function value at `x`. The
        values returned are the objective function value followed by the 
        gradient function value. 
    """
    function objgrad($(args...)) where {T}
        return NLPModels.objgrad(progData, x)
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the Hessian function value at `x`.
    """
    function hess($(args...)) where {T}
        return NLPModels.hess(progData, x)
    end
end

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

args_pre = [
    :(progData::CUTEstModel{T}),
    :(preComp::PrecomputeCUTEst{T}),
    :(x::Vector{T})
]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at the value `x`.
    """
    function obj($(args_pre...)) where {T}
        return NLPModels.obj(progData, x)
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`.
    """
    function grad($(args_pre...)) where {T}
        return NLPModels.grad(progData, x)
    end

    @doc """
         objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the objective function and gradient function value at `x`. The
        values returned are the objective function value followed by the 
        gradient function value. 
    """
    function objgrad($(args_pre...)) where {T}
        return NLPModels.objgrad(progData, x)
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the Hessian function value at `x`.
    """
    function hess($(args_pre...)) where {T}
        return NLPModels.hess(progData, x)
    end
end

###############################################################################
# Operations that are in-place. Makes use of precomputed values. 
###############################################################################

args_store = [
    :(progData::CUTEstModel{T}),
    :(preComp::PrecomputeCUTEst{T}),
    :(store::AllocateCUTEst{T}),
    :(x::Vector{T})
]


@eval begin 

    @doc """
        obj(
            $(join(string.(args_store),"\n\t    "))
        ) where {T,S}

    Computes the objective function at the value `x`.
    """
    function obj($(args_store...)) where {T}
        return NLPModels.obj(progData, x)
    end

    @doc """
        grad!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`. The gradient is computed in
        place and saved in `store.grad`.
    """
    function grad!($(args_store...)) where {T}
        NLPModels.grad!(progData, x, store.grad)
    end

    @doc """
         objgrad!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}

    Computes the objective function and gradient function value at `x`. The
        objective function value is returned. The gradient function value is 
        stored in `store.grad`. 
    """
    function objgrad!($(args_store...)) where {T}
        f,_ = NLPModels.objgrad!(progData, x, store.grad)
        return f
    end


    @doc """
        hess!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}
    
    Computes the Hessian function value at `x`. The Hessian function value is
        stored in `store.hess`.
    """
    function hess!($(args_store...)) where {T}
        store.hess .= NLPModels.hess(progData, x)
    end


end