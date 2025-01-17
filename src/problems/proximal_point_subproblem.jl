# Date: 01/17/2025
# Author: Christian Varner
# Purpose: Create structure for the proximal point subproblem

"""
"""
mutable struct ProximalPointSubproblem{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    progData::P1 where P1 <: AbstractNLPModel{T, S}
    progData_precomp::P2 where P2 <: AbstractPrecompute{T}
    progData_store::P3 where P3 <: AbstractProblemAllocate{T}
    penalty::T
    θkm1::S
end
function ProximalPointSubproblem(
    ::Type{T};
    progData::P1 where P1 <: AbstractNLPModel{T, S},
    progData_precomp::P2 where P2 <: AbstractPrecompute{T},
    progData_store::P3 where P3 <: AbstractProblemAllocate{T},
    penalty::T,
    θkm1::S
) where {T, S}

    progData.meta.nvar

    # initialize meta for proximal poitn
    meta = NLPModelMeta(
        progData.meta.nvar,
        name = "Proximal Point Subproblem: "*progData.meta.name,
        x0 = θkm1
    )

    # initialize
    counters = Counters()

    return ProximalPointSubproblem(meta, counters, progData, progData_precomp, 
        progData_store, penalty, θkm1)
end

# precomp data structure
"""
"""
struct PrecomputeProximalPointSubproblem{T} <: AbstractPrecompute{T}
    precomp::P where P <: AbstractPrecompute{T}
end
function PrecomputeProximalPointSubproblem(
    progData::ProximalPointSubproblem{T, S}
) where {T, S}
    return PrecomputeProximalPointSubproblem{T}(progData.progData_precomp)
end

# allocate data structure
"""
"""
struct AllocateProximalPointSubproblem{T} <: AbstractProblemAllocate{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocateProximalPointSubproblem(
    progData::ProximalPointSubproblem{T, S}
) where {T, S}
end

# initialize
"""
"""
function initialize(
    progData::ProximalPointSubproblem{T, S}
)
    precomp = PrecomputeProximalPointSubproblem(progData)
    store = AllocateProximalPointSubproblem(progData)
    
    return precomp, store
end

# Functionality

## operations without precomputed and allocated memory

args = [
    :(progData::ProximalPointSubproblem{T, S}),
    :(x::Vector{T})
]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))    
        ) where {T,S}

    Computes the objective function at the value `x`.
    """
    function NLPModels.obj($(args...)) where {T,S}
        increment!(progData, :neval_obj)
        return obj(progData.progData, x) + 
            .5 * progData.penalty * norm(x - progData.θkm1)^2
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient of the objective function at `x`. 
    """
    function NLPModels.grad($(args...)) where {T,S}
        increment!(progData, :neval_grad)
        return grad(progData.progData, x) + 
            progData.penalty .* (x - progData.θkm1)
    end

    @doc """
        objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at `x`, and gradient of the objective 
        function at `x`.
    """
    function NLPModels.objgrad($(args...)) where {T, S}
        o = obj(progData, x)
        g = grad(progData, x)
        return o, g
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
        
    Computes the Hessian of the objective function at `x`. 
    """
    function hess($(args...)) where {T,S}
        increment!(progData, :neval_hess)
        d = length(x)
        H = hess(progData.progData, x)
        for i in 1:d
            H[i,i] += progData.penalty
        end 
        return H
    end
end

## operation with precomputed but not allocated memory

args_precomp = [
    :(progData::ProximalPointSubproblem{T, S}),
    :(precomp::PrecomputeProximalPointSubproblem{T}),
    :(x::Vector{T})
]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))    
        ) where {T,S}

    Computes the objective function at the value `x`.
    """
    function NLPModels.obj($(args_precomp...)) where {T,S}
        increment!(progData, :neval_obj)
        return obj(progData.progData, progData.progData_precomp, x) + 
            .5 * progData.penalty * norm(x - progData.θkm1)^2
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient of the objective function at `x`. 
    """
    function NLPModels.grad($(args_precomp...)) where {T,S}
        increment!(progData, :neval_grad)
        return grad(progData.progData, progData.progData_precomp, x) + 
            progData.penalty .* (x - progData.θkm1)
    end

    @doc """
        objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at `x`, and gradient of the objective 
        function at `x`.
    """
    function NLPModels.objgrad($(args_precomp...)) where {T, S}
        o = obj(progData, precomp, x)
        g = grad(progData, precomp, x)
        return o, g
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
        
    Computes the Hessian of the objective function at `x`. 
    """
    function hess($(args_precomp...)) where {T,S}
        increment!(progData, :neval_hess)
        d = length(x)
        H = hess(progData.progData, progData.progData_precomp, x)
        for i in 1:d
            H[i,i] += progData.penalty
        end 
        return H
    end
end

## operation with precomputed and allocated memory

args_store = [
    :(progData::ProximalPointSubproblem{T, S}),
    :(precomp::PrecomputeProximalPointSubproblem{T}),
    :(store::AllocateProximalPointSubproblem{T}),
    :(x::Vector{T})
]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))    
        ) where {T,S}

    Computes the objective function at the value `x`.
    """
    function NLPModels.obj($(args_store...)) where {T,S}
        increment!(progData, :neval_obj)
        return obj(progData.progData, progData.progData_precomp, 
            progData.progData_store, x) + 
            .5 * progData.penalty * norm(x - progData.θkm1)^2
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient of the objective function at `x`. 
    """
    function NLPModels.grad!($(args_store...)) where {T,S}
        increment!(progData, :neval_grad)

        grad!(progData.progData, progData.progData_precomp, 
            progData.progData_store, x)

        store.grad .= progData.progData_store.grad 
            + progData.penalty .* (x - progData.θkm1) 
    end

    @doc """
        objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at `x`, and gradient of the objective 
        function at `x`.
    """
    function NLPModels.objgrad!($(args_store...)) where {T, S}
        o = obj(progData, precomp, store, x)
        g = grad(progData, precomp, store, x)
        return o, g
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
        
    Computes the Hessian of the objective function at `x`. 
    """
    function hess!($(args_store...)) where {T,S}
        increment!(progData, :neval_hess)

        hess!(progData.progData, progData.progData_precomp, 
            progData.progData_store, x)
        
        store.hess .= progData.progData_store.hess
        d = length(x)
        for i in 1:d
            store += progData.penalty
        end 
    end
end