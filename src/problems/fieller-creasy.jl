# Date: 2025/02/09
# Authors: Christian Varner
# Purpose Implementation of the fieller-creasy problem

"""
"""
mutable struct FiellerCreasy{T, S} <: AbstractDefaultQL{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    y1::Vector{T}
    y2::Vector{T}
    σ::Float64
end
function FiellerCreasy(
    ::Type{T};
    μ::Vector{Float64},
    σ::Float64 = 1.,
    nobs::Int64 = 100,
    θ::Float64 = 2.,
) where {T}

    # initialize the meta data and counters
    meta = NLPModelMeta(
        1,
        name = "Quasi-likelihood with logistic link function and centered exp",
        x0 = zeros(T, 1)
    )
    counters = Counters()

    # generate observations
    y1 = σ .* randn(T, nobs) .+ μ
    y2 = σ .* randn(T, nobs) .+ μ/θ

    return FiellerCreasy(meta, counters, y1, y2, σ)
end
function FiellerCreasy(
    y1::Vector{T},
    y2::Vector{T},
    σ::Float64
) where {T}
    # initialize the meta data and counters
    meta = NLPModelMeta(
        1,
        name = "Quasi-likelihood with logistic link function and centered exp",
        x0 = zeros(T, 1)
    )
    counters = Counters()
    return FiellerCreasy(meta, counters, y1, y2, σ)
end

"""
"""
struct PrecomputeFiellerCreasy{T} <: AbstractPrecompute{T}
end

"""
"""
struct AllocateFiellerCreasy{T} <: AbstractProblemAllocate{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocateFiellerCreasy(progData::FiellerCreasy{T, S}) where {T, S}
    grad = zeros(T, 1)
    hess = zeros(T, 1, 1)
    return AllocateFiellerCreasy{T}(grad, hess)
end

"""
"""
function initialize(progData::FiellerCreasy{T, S}) where {T, S}
    precompute = PrecomputeFiellerCreasy{T}()
    store = AllocateFiellerCreasy(progData)

    return precompute, store
end

###############################################################################
# Operations that are not in-place. Does not make use of precomputed values.
###############################################################################

args = [:(progData::FiellerCreasy{T, S}), 
        :(x::Vector{T})
       ]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at the value `x`.

        -(z * ((y_2^2 - y_1^2) * z - 2 * y_1 * y_2)) / (2 * (z^2 + 1) * σ^2)
    """
    function NLPModels.obj($(args...)) where {T, S}
        increment!(progData, :neval_obj)
        obj = 0
        for i in 1:length(progData.y1)
            t1 = (progData.y2[i]^2 - progData.y1[i]^2) .* x .- 
                (2 * progData.y1[i] * progData.y2[i]) 
            t2 = (2 .* (1 .+ x .^ 2 ) .* progData.σ^2) 
            obj += (t1 ./ t2) .* x
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
        for i in 1:length(progData.y1)
            t1 = (progData.y2[i] .+ progData.y1[i] .* x)
            t2 = (progData.y1[i] .- progData.y2[i] .* x) 
            g .-= (t1 .* t2) ./ (progData.σ^2 .* (1 .+ x .^ 2) .^ 2)
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

    -(2 * y_1 * y_2 * x^3 + (3 * y_2^2 - 3 * y_1^2) * x^2 - 6 * y_1 * y_2 * x - y_2^2 + y_1^2) / (σ^2 * (x^2 + 1)^3)
    """
    function hess($(args...)) where {T, S}
        increment!(progData, :neval_hess)
        H = zeros(T, 1, 1)
        for i in 1:length(progData.y1)
            t1 = (2 * progData.y1[i] * progData.y2[i]) .* (x .^ 3)
            t2 = (3 * progData.y2[i]^2 - 3 * progData.y1[i]^2) .* (x.^2)
            t3 = (6 * progData.y1[i] * progData.y2[i]) .* x 
            t4 = progData.y2[i]^2 - progData.y1[i]^2
            t5 = (progData.σ^2 .* (1 .+ x.^2).^3)
            H .-= (t1 + t2 - t3 - t4) ./ t5
        end
        return H
    end

end # end eval

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

args_pre = [
    :(progData::FiellerCreasy{T,S}),
    :(preComp::PrecomputeFiellerCreasy{T}),
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
    :(progData::FiellerCreasy{T,S}),
    :(preComp::PrecomputeFiellerCreasy{T}),
    :(store::AllocateFiellerCreasy{T}),
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
        for i in 1:length(progData.y1)
            t1 = (progData.y2[i]^2 - progData.y1[i]^2) .* x .- 
                (2 * progData.y1[i] * progData.y2[i]) 
            t2 = (2 .* (1 .+ x .^ 2 ) .* progData.σ^2) 
            obj += (t1 ./ t2) .* x
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
        for i in 1:length(progData.y1)
            t1 = (progData.y2[i] .+ progData.y1[i] .* x)
            t2 = (progData.y1[i] .- progData.y2[i] .* x) 
            store.grad .-= (t1 .* t2) ./ (progData.σ^2 .* (1 .+ x .^ 2) .^ 2)
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
        for i in 1:length(progData.y1)
            t1 = (2 * progData.y1[i] * progData.y2[i]) .* (x .^ 3)
            t2 = (3 * progData.y2[i]^2 - 3 * progData.y1[i]^2) .* (x.^2)
            t3 = (6 * progData.y1[i] * progData.y2[i]) .* x 
            t4 = progData.y2[i]^2 - progData.y1[i]^2
            t5 = (progData.σ^2 .* (1 .+ x.^2).^3)
            store.hess .-= (t1 + t2 - t3 - t4) ./ t5
        end
    end

    """
    """
    function fisher!($(args_store...); recompute = true) where {T, S}
        increment!(progData, :neval_hess)
        fill!(store.hess, 0) 
        for i in 1:length(progData.y1)
            t1 = (progData.y2[i] .+ progData.y1[i] .* x) .^ 2
            t2 = progData.σ^2 .* (1 + x .^ 2) .^ 3 
            store.hess .-= (t1 ./ t2)
        end
    end
end