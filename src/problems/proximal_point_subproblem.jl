# Date: 01/17/2025
# Author: Christian Varner
# Purpose: Create structure for the proximal point subproblem

# IMPLEMENTATION NOTES: EXTENSIONS
#
# To extend this to bregman distance function, 
#   1) add an extra field for distance functions
#   2) distance functions can be implemented in seperate files with
#       corresponding derivative + hessian information (like the link functions)

"""
    ProximalPointSubproblem{T, S} <: AbstractNLPModel{T, S}

Mutable struct that represents a proximal point subproblem.

# Objective Function

Let ``f(\\theta)`` be the objective function of another optimization problem.
When using the proximal point gradient method, the following
subproblem is formulated and solved to obtain the next iterate;
for a penalty term ``\\lambda \\in \\mathbb{R}_{\\geq 0}``
```math
    \\theta_{k+1} = 
    \\arg\\min_{\\theta} f(\\theta) + \\frac{\\lambda}{2}||\\theta - \\theta_{k}||_2^2
```
where ``\\theta_{k}`` is the current iterate, and ``||\\cdot||_2`` is the L2 norm.
This struct represents this optimization problem for finding ``\\theta_{k+1}``.
In particular, ``F(\\theta)`` is
```math
F(\\theta) = f(\\theta) + \\frac{\\lambda}{2}||\\theta - \\theta_{k}||_2^2
```

# Fields

- `meta::NLPModelMeta{T, S}`, NLPModel struct for storing meta information for 
    the problem
- `counters::Counters`, NLPModel Counter struct that provides evaluations 
    tracking.
- `progData::P1 where P1 <: AbstractNLPModel{T, S}`, problem data from
    another optimization problem for which a proximal subproblem is to
    be created. In language of the objective function description, 
    `progData` represents ``f(\\theta)``.
- `progData_precomp::P2 where P2 <: AbstractPrecompute{T}`, precomputed
    data structure for the optimization problem represented by `progData`.
    In the language of the objective function description, this is
    the precomputed values for the implementation of ``f(\\theta)``.
- `progData_store::P3 where P3 <: AbstractProblemAllocate{T}`, storage
    data structure for the optimization problem represented by `progData`.
    In the language of the objective function description, this is the
    storage data structure for the implementation of ``f(\\theta)``.
- `penalty::T`, penalty term applied to the distance term for the proximal
    point subproblem.
- `θkm1::S`, parameter that is part of the distance function.

!!! note
    Currently, the proximal point subproblem only uses the L2 norm as the penalty
    function. This is part of a more general class of methods called Bregman-
    Distance gradient methods.

# Constructors

    function ProximalPointSubproblem(::Type{T};
        progData::P1 where P1 <: AbstractNLPModel{T, S},
        progData_precomp::P2 where P2 <: AbstractPrecompute{T},
        progData_store::P3 where P3 <: AbstractProblemAllocate{T}, penalty::T,
        θkm1::S) where {T, S}

Construct the `struct` for the Proximal Point Subproblem. This will create the
    `meta::NLPModelMeta` and the `counters::Counters` data structures, and
    return a structure of type `ProximalPointSubproblem{T, S}`

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
    PrecomputeProximalPointSubproblem{T} <: AbstractPrecompute{T}

Store precomputed values for the proximal point subproblem.
"""
struct PrecomputeProximalPointSubproblem{T} <: AbstractPrecompute{T} end

# allocate data structure
"""
    AllocateProximalPointSubproblem{T} <: AbstractProblemAllocate{T}

Allocate memory for the proximal point subproblem. 

# Fields

- `grad::Vector{T}`, memory for the gradient of the objective
- `hess::Matrix{T}`, memory for the hessian of the objective

# Constructor

    AllocateProximalPointSubproblem(progData::ProximalPointSubproblem{T, S}) 
        where {T, S}

Requests the memory for each of the fields and initializes each buffer to
    contain zeros. Returns the structure.
"""
struct AllocateProximalPointSubproblem{T} <: AbstractProblemAllocate{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocateProximalPointSubproblem(
    progData::ProximalPointSubproblem{T, S}
) where {T, S}
    d = progData.meta.nvar
    
    return AllocateProximalPointSubproblem(
        zeros(T, d),
        zeros(T, d, d)
    )
end

# initialize
"""
    initialize(progData::ProximalPointSubproblem{T, S})

Constructs `ProximalPointSubproblem{T, S}` and 
`AllocateProximalPointSubproblem{T}`, returning them.

# Arguments

- `progData::ProximalPointSubproblem{T, S}`, problem data that represents
    the problem.
"""
function initialize(
    progData::ProximalPointSubproblem{T, S}
) where {T, S}
    precomp = PrecomputeProximalPointSubproblem{T}()
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

    Computes the gradient of the objective function at `x`. This value is
        stored in `store.grad`. 
    """
    function NLPModels.grad!($(args_store...)) where {T,S}
        increment!(progData, :neval_grad)

        grad!(progData.progData, progData.progData_precomp, 
            progData.progData_store, x)

        store.grad .= progData.progData_store.grad + 
            progData.penalty .* (x - progData.θkm1) 
    end

    @doc """
        objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at `x`, and gradient of the objective 
        function at `x`. The gradient is stored in `store.grad`.
    """
    function NLPModels.objgrad!($(args_store...)) where {T, S}
        o = obj(progData, precomp, store, x)
        grad!(progData, precomp, store, x)
        return o
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
        
    Computes the Hessian of the objective function at `x`. The hessian is
        stored in `store.hess`.
    """
    function hess!($(args_store...)) where {T,S}
        increment!(progData, :neval_hess)

        hess!(progData.progData, progData.progData_precomp, 
            progData.progData_store, x)
        
        store.hess .= progData.progData_store.hess
        d = length(x)
        for i in 1:d
            store.hess[i, i] += progData.penalty
        end 
    end
end