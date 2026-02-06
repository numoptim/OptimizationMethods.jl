# Date: 2025/10/20
# Author: Christian Varner
# Purpose: Implement the variance function objective

"""
"""
mutable struct SOSVarianceObjective{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    response::Vector{T}
    μ::Vector{T}
    variance::Function
    variance_first_derivative::Function
    variance_second_derivative::Function
end
function SOSVarianceObjective(
    μ::Vector{T},
    response::Vector{T},
    V::Function,
    dV::Function,
    ddV::Function,
    x0::Vector{T}
) where {T}

    nvar = length(x0)
    meta = NLPModelMeta(nvar, name = "Variance Function Objective", x0 = x0)

    return SOSVarianceObjective{T, Vector{T}}(
        meta,
        Counters(),
        response,
        μ,
        V,
        dV,
        ddV,
    )
end

"""
"""
struct PrecomputeVarObj{T} <: AbstractPrecompute{T}
end

"""
"""
struct AllocateVarObj{T} <: AbstractProblemAllocate{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocateVarObj(progData::SOSVarianceObjective{T,S}) where {T,S}
    nvar = length(progData.meta.x0)
    grad = zeros(T, nvar)
    hess = zeros(T, nvar, nvar)
    return AllocateVarObj{T}(grad, hess)
end

"""
"""
function initialize(progData::SOSVarianceObjective{T,S}) where {T,S}
    precompute = PrecomputeVarObj{T}()
    store = AllocateVarObj(progData)

    return precompute, store
end


###############################################################################
# Operations that are not in-place. Does not make use of precomputed values.
###############################################################################

args = [:(progData::SOSVarianceObjective{T, S}), 
        :(x::Vector{T})
       ]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at the value `x`.
    """
    function NLPModels.obj($(args...)) where {T, S}
        increment!(progData, :neval_obj)
        obj = 0
        for i in 1:length(progData.response)
            obj += .5 * ((progData.response[i] - progData.μ[i]) ^ 2 - 
            progData.variance(progData.μ[i], x))^2
        end
        return obj
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`.
    """
    function NLPModels.grad($(args...)) where {T, S}
        increment!(progData, :neval_grad)
        g = zeros(T, length(x))
        for i in 1:length(progData.response)
            g .-= ((progData.response[i] - progData.μ[i]) ^ 2 - 
            progData.variance(progData.μ[i], x)) * progData.variance_first_derivative(progData.μ[i], x)
        end
        return g
    end

    @doc """
         objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the objective function and gradient function value at `x`. The
        values returned are the objective function value followed by the 
        gradient function value. 
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
    
    Computes the Hessian function value at `x`.
    """
    function hess($(args...)) where {T, S}
        increment!(progData, :neval_hess)
        H = zeros(T, length(x), length(x))
        for i in 1:length(progData.response)
            v = progData.variance(progData.μ[i], x)
            dv = progData.variance_first_derivative(progData.μ[i], x) 
            ddv = progData.variance_second_derivative(progData.μ[i], x) 
            t1 = dv * dv'
            t2 = v * ddv
            t3 = (progData.response[i] - progData.μ[i]) ^ 2 * ddv
            H .+= t1 + t2 - t3 
        end
    end

end # end eval

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

args_pre = [
    :(progData::SOSVarianceObjective{T,S}),
    :(preComp::PrecomputeVarObj{T}),
    :(x::Vector{T})
]

funcs = Dict(
    :(NLPModels.obj) => "Computes the objective function value at `x`.",
    :(NLPModels.grad) => "Computes the gradient function value at `x`.",
    :(NLPModels.objgrad) => "Computes the objective and gradient at `x`. 
        Returns the objective value, then the gradient value.",
    :(hess) => "Computes the Hessian value at `x`."
)

for (func, desc) in funcs
    
    #Precompute does not have any additional information.
    #This loop will call the version of the function that uses only 
    #progData and x.
    @eval begin
        @doc """
            $(string($func))(
                $(join(string.(args_pre),",\n\t    "))
            ) where {T,S}
        
        $($desc)
        """
        function $func($(args_pre...)) where {T,S}
            return $func(progData, x)
        end

    end
end 

###############################################################################
# Operations that are in-place. Makes use of precomputed values. 
###############################################################################

args_store = [
    :(progData::SOSVarianceObjective{T,S}),
    :(preComp::PrecomputeVarObj{T}),
    :(store::AllocateVarObj{T}),
    :(x::Vector{T})
]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at the value `x`.
        If `recompute = false`, then values already stored in `store` are 
        used in the computation, otherwise the necessary values are recomputed
        and used. 
    """
    function NLPModels.obj($(args_store...); recompute = true) where {T, S}
        increment!(progData, :neval_obj)
        obj = 0
        for i in 1:length(progData.response)
            obj += .5 * ((progData.response[i] - progData.μ[i]) ^ 2 - 
            progData.variance(progData.μ[i], x))^2
        end
        return obj 
    end

    @doc """
        grad!(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`.
        Stores the computed gradient vector into `store.grad`. If 
        `recompute = false` then values that are already in `store`` are used
        for computation. Otherwise, values are recomputed and used.
    """
    function NLPModels.grad!($(args_store...); recompute = true) where {T, S}
        increment!(progData, :neval_grad)
        fill!(store.grad, 0)
        for i in 1:length(progData.response)
            store.grad .-= ((progData.response[i] - progData.μ[i]) ^ 2 - 
            progData.variance(progData.μ[i], x)) * progData.variance_first_derivative(progData.μ[i], x)
        end
    end

    @doc """
         objgrad!(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the objective function and gradient function value at `x`. The
        values returned are the objective function and the gradient is
        stored in `store.grad`. If `recompute = false`, then values already 
        in `store` are used for computation, otherwise values required in the
        computation are computed and used.
    """
    function NLPModels.objgrad!($(args_store...); recompute = true) where {T, S}
        NLPModels.grad!(progData, precomp, store, x; recompute = recompute)
        o = NLPModels.obj(progData, precomp, store, x; recompute = false)
        return o
    end

    @doc """
        hess!(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the Hessian function value at `x`.
        Utilizes the precomputed values in `precomp` and stores the result in
        `store.hess`. If `recompute = false`, tries to compute the hessian
        with values already stored in `store`, otherwise recomputes the 
        necessary quantities and computes the hessian.
    """
    function hess!($(args_store...); recompute = true) where {T, S}
        increment!(progData, :neval_hess)
        fill!(store.hess, 0)
        for i in 1:length(progData.response)
            v = progData.variance(progData.μ[i], x)
            dv = progData.variance_first_derivative(progData.μ[i], x) 
            ddv = progData.variance_second_derivative(progData.μ[i], x) 
            t1 = dv * dv'
            t2 = v * ddv
            t3 = (progData.response[i] - progData.μ[i]) ^ 2 * ddv              
            store.hess .+= t1 + t2 - t3
        end
    end

    """
    """
    function fisher!($(args_store...); recompute = true) where {T, S}
        increment!(progData, :neval_hess)
        fill!(store.hess, 0)
        for i in 1:length(progData.response)
            v = progData.variance(progData.μ[i], x)
            dv = progData.variance_first_derivative(progData.μ[i], x) 
            ddv = progData.variance_second_derivative(progData.μ[i], x) 
            t1 = dv * dv'
            t2 = v * ddv
            store.hess .+= t1 + t2                                              
        end
    end
end
