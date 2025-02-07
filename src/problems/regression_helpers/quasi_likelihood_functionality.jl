# Date: 01/06/2025
# Author: Christian Varner
# Purpose: Implementation of quasi-likelihood functionality
# This includes functions to compute the objective, gradient, and
# hessian for an arbitrary mean and variance function

###############################################################################
# Operations that are not in-place. Does not make use of precomputed values.
###############################################################################

args = [:(progData::P where P<:AbstractDefaultQL{T, S}), 
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
        η = progData.design * x
        μ_hat = progData.mean.(η)
        obj = 0
        for i in 1:length(progData.response)
            # ## create numerical integration problem
            # prob = IntegralProblem(
            #     progData.weighted_residual, 
            #     (0, μ_hat[i]), 
            #     progData.response[i]
            #     )

            # ## solve the numerical integration problem
            # ## TODO: this is just with some default parameters
            # ## TODO: what happens when the code is false
            # obj -= solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u

            obj -= quadgk(
                x -> progData.weighted_residual(x, progData.response[i]),
                0, μ_hat[i])[1]
        end

        return T(obj)
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`.
    """
    function NLPModels.grad($(args...)) where {T, S}
        increment!(progData, :neval_grad)

        # compute values required for gradient
        η = progData.design * x
        μ_hat = progData.mean.(η)
        ∇μ_η = progData.mean_first_derivative.(η)
        residual = progData.weighted_residual.(μ_hat, progData.response) 

        # compute and return gradient
        g = zeros(T, length(x))
        for i in 1:length(progData.response)
            g .-= residual[i] * ∇μ_η[i] .* view(progData.design, i, :)
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

        # compoute required quantities
        η = progData.design * x
        μ_hat = progData.mean.(η)
        ∇μ_η = progData.mean_first_derivative.(η)
        ∇∇μ_η = progData.mean_second_derivative.(η)
        var = progData.variance.(μ_hat)
        ∇var = progData.variance_first_derivative.(μ_hat)
        r = progData.weighted_residual.(μ_hat, progData.response)

        # compute hessian
        nobs, nvar = size(progData.design)
        H = zeros(T, nvar, nvar)
        for i in 1:nobs
            t1 = var[i]^(-1) * ∇var[i] * (∇μ_η[i]^2) * r[i]
            t2 = var[i]^(-1) * (∇μ_η[i]^2)
            t3 = r[i] * ∇∇μ_η[i] 
            oi = view(progData.design, i, :) * view(progData.design, i, :)'

            H .-= (t3 - t1 - t2) .* oi 
        end

        return H
    end

end

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

args_pre = [
    :(progData::P where P<:AbstractDefaultQL{T, S}),
    :(precomp::P where P<:AbstractDefaultQLPrecompute{T}),
    :(x::Vector{T})
]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at the value `x`.
    """
    function NLPModels.obj($(args_pre...)) where {T, S}
        return NLPModels.obj(progData, x)
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`.
    """
    function NLPModels.grad($(args_pre...)) where {T, S}
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
    function NLPModels.objgrad($(args_pre...)) where {T, S}
        return NLPModels.objgrad(progData, x)
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the Hessian function value at `x`.
        Utilizes the precomputed values in `precomp`.
    """
    function hess($(args_pre...)) where {T, S}
        increment!(progData, :neval_hess)

        # compoute required quantities
        η = progData.design * x
        μ_hat = progData.mean.(η)
        ∇μ_η = progData.mean_first_derivative.(η)
        ∇∇μ_η = progData.mean_second_derivative.(η)
        var = progData.variance.(μ_hat)
        ∇var = progData.variance_first_derivative.(μ_hat)
        r = progData.weighted_residual.(μ_hat, progData.response)

        # compute hessian
        nobs, nvar = size(progData.design)
        H = zeros(T, nvar, nvar)
        for i in 1:nobs
            t1 = var[i]^(-1) * ∇var[i] * (∇μ_η[i]^2) * r[i]
            t2 = var[i]^(-1) * (∇μ_η[i]^2)
            t3 = r[i] * ∇∇μ_η[i] 

            H .-= (t3 - t1 - t2) .* view(precomp.obs_obs_t, i, :, :)
        end

        return H
    end
end

###############################################################################
# Operations that are in-place. Makes use of precomputed values. 
###############################################################################

args_store = [
    :(progData::P where P<:AbstractDefaultQL{T, S}),
    :(precomp::P where P<:AbstractDefaultQLPrecompute{T}),
    :(store::P where P<:AbstractDefaultQLAllocate{T}),
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
        
        # recompute possible values
        if recompute
            store.linear_effect .= progData.design * x
            store.μ .= progData.mean.(store.linear_effect)
        end
        
        # recompute possible objective functions
        obj = 0
        for i in 1:length(progData.response)
            # ## create numerical integration problem
            # prob = IntegralProblem(
            #     progData.weighted_residual, 
            #     (0, store.μ[i]), 
            #     progData.response[i]
            #     )

            # ## solve the numerical integration problem
            # ## TODO: this is just with some default parameters
            # ## TODO: what happens when the code is false
            # obj -= solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u

            obj -= quadgk(
                x -> progData.weighted_residual(x, progData.response[i]),
                0, store.μ[i])[1]
        end

        return obj
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`.
        Stores the computed gradient vector into `store.grad`. If 
        `recompute = false` then values that are already in `store`` are used
        for computation. Otherwise, values are recomputed and used.
    """
    function NLPModels.grad!($(args_store...); recompute = true) where {T, S}
        increment!(progData, :neval_grad)
        if recompute
            store.linear_effect .= progData.design * x
            store.μ .= progData.mean.(store.linear_effect)
            store.∇μ_η .= progData.mean_first_derivative.(store.linear_effect)
            store.variance .= progData.variance.(store.μ)
            store.weighted_residual .= progData.weighted_residual.(store.μ, progData.response)
        end

        fill!(store.grad, 0)
        for i in 1:length(progData.response)
            store.grad .-= store.weighted_residual[i] * store.∇μ_η[i] .* 
                view(progData.design, i, :)
        end
    end

    @doc """
         objgrad(
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
        hess(
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

        # compoute required quantities
        if recompute
            store.linear_effect .= progData.design * x
            store.μ .= progData.mean.(store.linear_effect)
            store.∇μ_η .= progData.mean_first_derivative.(store.linear_effect)
            store.∇∇μ_η .= progData.mean_second_derivative.(store.linear_effect)
            store.variance .= progData.variance.(store.μ)
            store.∇variance .= progData.variance_first_derivative.(store.μ)
            store.weighted_residual .= 
                progData.weighted_residual.(store.μ, progData.response)
        end

        # compute hessian
        nobs = size(progData.design, 1)
        fill!(store.hess, 0)
        for i in 1:nobs
            t1 = store.variance[i]^(-1) * store.∇variance[i] * 
                (store.∇μ_η[i]^2) * store.weighted_residual[i]
            t2 = store.variance[i]^(-1) * (store.∇μ_η[i]^2)
            t3 = store.weighted_residual[i] * store.∇∇μ_η[i] 

            store.hess .-= (t3 - t1 - t2) .* view(precomp.obs_obs_t, i, :, :)
        end
    end

end
