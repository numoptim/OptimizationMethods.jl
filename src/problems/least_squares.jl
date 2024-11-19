# OptimizationMethods.jl

"""
    LeastSquares{T,S} <: AbstractNLSModel{T,S}

Implements a least squares problem where the coefficient matrix's and constant 
    vector's entries are independent Gaussian random variables.

# Objective Function

```math
\\min_{x} 0.5\\Vert F(x) \\Vert_2^2,
```
where
```math
F(x) = A * x - b.
```

`A` is the coefficient matrix. `b` is the constant vector.

# Fields 

- `meta::NLPModelMeta{T, S}`, data structure for nonlinear programming models
- `nls_meta::NLSMeta{T, S}`, data structure for nonlinear least squares models
- `counters::NLSCounters`, counters for nonlinear least squares models
- `coef::Matrix{T}`, coefficient matrix, `A`, for least squares problem 
- `cons::Vector{T}`, constant vector, `b`, for least squares problem

# Constructors

    LeastSquares(::Type{T}; nequ=1000, nvar=50) where {T}

Constructs a least squares problems with `1000` equations and `50` unknowns,
    where the entries of the matrix and constant vector are independent
    standard Gaussian variables.

## Arguments
    
- `T::DataType`, specific data type of the optimization parameter

## Optional Keyword Arguments
    
- `nequ::Int64=1000`, the number of equations in the system 
- `nvar::Int64=50`, the number of parameters in the system 

"""
mutable struct LeastSquares{T, S} <: AbstractNLSModel{T, S}
    meta::NLPModelMeta{T, S}
    nls_meta::NLSMeta{T, S}
    counters::NLSCounters
    coef::Matrix{T}
    cons::Vector{T}

    LeastSquares{T, S}(meta, nls_meta, counters, coef, cons) where {T, S} =
    begin
        @assert size(coef, 1) == length(cons) "Number of responses is not 
        equal to number of rows in `coef`"
        return new(meta, nls_meta, counters, coef, cons)
    end
end
function LeastSquares(
    ::Type{T};
    nequ::Int64 = 1000, 
    nvar::Int64 = 50
) where {T}

    meta = NLPModelMeta(
        nvar, 
        name = "Least Squares",
        x0 = ones(T, nvar),
    )

    nls_meta = NLSMeta{T, Vector{T}}(
        nequ,
        nvar,
        nnzj = nequ * nvar,
        nnzh = nvar * nvar,
        lin = collect(1:nequ),
    )

    return LeastSquares{T, Vector{T}}(
        meta, 
        nls_meta, 
        NLSCounters(),
        randn(T, nequ, nvar),
        randn(T, nequ),
    )
end
function LeastSquares(
    design::Matrix{T},
    response::Vector{T};
    x0 = ones(T, size(design, 2))
) where {T}

    nequ, nvar = size(design)
    @assert nvar == length(x0) "Number of columns in `design` is not
    equal to number of entires in `x0`"

    meta = NLPModelMeta(
        nvar = nvar,
        name = "Least Squares",
        x0 = x0, 
    )

    nls_meta = NLSMeta{T, Vector{T}}(
        nequ,
        nvar,
        nnzj = nequ * nvar,
        nnzh = nvar * nvar,
        lin = collect(1:nequ),
    )

    return LeastSquares{T, Vector{T}}(
        meta,
        nls_meta,
        NLSCounters(),
        design,
        response
    )
end

"""
    PrecomputeGLS{T} <: AbstractPrecompute{T}

Immutable structure for initializing and storing repeatedly used calculations
    related to coefficient matrix `A` and constant vector `b`.
    
# Fields 

- `coef_t_coef::Matrix{T}`, stores `A'*A`
- `coef_t_cons::Vector{T}`, stores `A'*b`
- `cons_t_cons::T`, stores `b'*b`

# Constructors

    PrecomputeGLS(prog::LeastSquares{T,S}) where {T,S}

Computes `A'*A`, `A'*b`, `b'*b` given a Gaussian Least Squares program, `prog`. 
"""
struct PrecomputeGLS{T} <: AbstractPrecompute{T}
    coef_t_coef::Matrix{T}
    coef_t_cons::Vector{T}
    cons_t_cons::T
end

function PrecomputeGLS(prog::LeastSquares{T,S}) where {T,S}
    coef_t_coef = prog.coef'*prog.coef
    coef_t_cons = prog.coef'*prog.cons 
    cons_t_cons = prog.cons'*prog.cons

    return PrecomputeGLS(coef_t_coef, coef_t_cons, cons_t_cons)
end

"""
    AllocateGLS{T} <: AbstractProblemAllocate{T}

Immutable structure for preallocating important quantities for the Gaussian
    Least Squares problem.

# Fields 

- `res::Vector{T}`, storage for residual vector
- `jac::Matrix{T}`, storage for jacobian matrix
- `grad::Vector{T}`, storage for gradient vector
- `hess::Matrix{T}`, storage for Hessian matrix 

# Constructors

    AllocateGLS(prog::LeastSquares{T,S}) where {T,S}
    AllocateGLS(
        prog::LeastSquares{T,S},
        preComp::precomputeGLS{T},
    ) where {T,S}

Preallocates data structures for Gaussian Least Squares based on optimization
    problem's data. The values of `r` and `grad` are set to zero of type `T`.
    `jac` and `hess` are set to `A` and `A'*A`. If `preComp` is supplied, then
    `hess` is set to `preComp.coef_t_coef`.
"""
struct AllocateGLS{T} <: AbstractProblemAllocate{T}
    res::Vector{T}
    jac::Matrix{T}
    grad::Vector{T}
    hess::Matrix{T}
end

function AllocateGLS(prog::LeastSquares{T,S}) where {T,S}

    return AllocateGLS(
        zeros(T, prog.nls_meta.nequ),
        prog.coef,
        zeros(T, prog.nls_meta.nvar),
        prog.coef' * prog.coef,
    )
end

function AllocateGLS(
    prog::LeastSquares{T,S},
    preComp::PrecomputeGLS{T}
) where {T,S}

    return AllocateGLS(
        zeros(T, prog.nls_meta.nequ),
        prog.coef,
        zeros(T, prog.nls_meta.nvar),
        preComp.coef_t_coef,
    )
end

"""
    initialize(progData::LeastSquares{T,S}) where {T,S}

Generates the precomputed and storage structs given a Gaussian Least Squares 
    problem.
"""
function initialize(progData::LeastSquares{T,S}) where {T,S}

    precompute = PrecomputeGLS(progData)
    store = AllocateGLS(progData, precompute)

    return precompute, store
end


###############################################################################
# Operations that are not in-place. Does not make use of precomputed values.
###############################################################################

args = [
    :(progData::LeastSquares{T,S}),
    :(x::Vector{T})
]

@eval begin 

    @doc """
        residual(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
        
    Computes the residual of at the value `x`. That is, `A * x - b`.
    """  
    function NLPModels.residual($(args...)) where {T,S}
        increment!(progData, :neval_residual)
        return progData.coef * x - progData.cons
    end

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))    
        ) where {T,S}

    Computes the objective function at the value `x`.
    """
    function NLPModels.obj($(args...)) where {T,S}
        r = residual(progData, x)
        increment!(progData, :neval_obj)
        return T(0.5 * dot(r, r))
    end

    @doc """
        jac_residual(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the jacobian of the residual function at `x`. This just returns the 
        coefficient matrix `A` (stored in `progData.coef`).
    """
    function NLPModels.jac_residual($(args...)) where {T,S}
        increment!(progData, :neval_jac_residual)
        return progData.coef
    end


   @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient of the objective function at `x`. This is `A'(A*x-b)`,
        which is equivalent to `J'*r` where `J` is the Jacobian and `r` is the 
        residual.
    """
    function NLPModels.grad($(args...)) where {T,S}
        r = residual(progData, x)
        J = jac_residual(progData, x)
        increment!(progData, :neval_grad)
        return J'*r
    end

    @doc """
        objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at `x`, and gradient of the objective function at `x`.
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
        
    Computes the Hessian of the objective function at `x`. This is `A'A`.
    """
    function hess($(args...)) where {T,S}
        increment!(progData, :neval_hess)
        return progData.coef'*progData.coef
    end
end

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

args_pre = [
    :(progData::LeastSquares{T,S}),
    :(preComp::PrecomputeGLS{T}),
    :(x::Vector{T})
]

@eval begin
   @doc """
        residual(
            $(join(string.(args_pre),",\n\t    "))
        ) where {T,S}
        
    Computes the residual of at the value `x`. That is, `A * x - b`.
    """
    function NLPModels.residual($(args_pre...)) where {T,S}
        return residual(progData, x)
    end

    @doc """
        obj(
            $(join(string.(args_pre),",\n\t    "))    
        ) where {T,S}

    Computes the objective function at the value `x`.
    """
    function NLPModels.obj($(args_pre...)) where {T,S}
        return obj(progData, x)
    end

    @doc """
        jac_residual(
            $(join(string.(args_pre),",\n\t    "))
        ) where {T,S}

    Computes the jacobian of the residual function at `x`. This just returns the 
        coefficient matrix `A` (stored in `progData.coef`).
    """
    function NLPModels.jac_residual($(args_pre...)) where {T,S}
        return jac_residual(progData, x)
    end

    @doc """
        grad(
            $(join(string.(args_pre),",\n\t    "))
        ) where {T,S}

    Computes the gradient of the objective function at `x`. This is `A'(A*x-b)`,
        which is equivalent to `A'A * x - A*b`
    """
    function NLPModels.grad($(args_pre...)) where {T,S}
        increment!(progData, :neval_grad)
        return preComp.coef_t_coef * x - preComp.coef_t_cons
    end

    @doc """
        objgrad(
            $(join(string.(args_pre),",\n\t    "))
        ) where {T,S}

    Compute the objective function at `x`, and computes the gradient of the objective at 'x'.
    The computation of the gradient uses the precomputed values `precomp.coef_t_coef` and
    `precomp.coef_t_cons`.
    """
    function NLPModels.objgrad($(args_pre...)) where {T, S}
        o = obj(progData, preComp, x)
        g = grad(progData, preComp, x)
        return o, g
    end

    @doc """
        hess(
            $(join(string.(args_pre),",\n\t    "))
        ) where {T,S}
        
    Computes the Hessian of the objective function at `x`. This is `A'A`. Returns
        the value `preComp.coef_t_coef` which contains this calculation.
    """
    function hess($(args_pre...)) where {T,S}
        increment!(progData, :neval_hess)
        return preComp.coef_t_coef
    end
end

###############################################################################
# Operations that are in-place. Makes use of precomputed values. 
###############################################################################

args_store = [
    :(progData::LeastSquares{T,S}),
    :(preComp::PrecomputeGLS{T}),
    :(store::AllocateGLS{T}),
    :(x::Vector{T})
]

@eval begin 
    
    @doc """
        residual!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}

    Computes the residual and updates it in `store.res`. `preComp` is not used.
    """
    function NLPModels.residual!($(args_store...)) where {T,S}
        increment!(progData, :neval_residual)
        store.res .= progData.coef * x - progData.cons
        return nothing
    end

    @doc """
        obj(
            $(join(string.(args_store),",\n\t    "));
            recompute::Bool=true
        ) where {T,S}
        
    Computes the objective function using the existing value in `store.res`. If
        `recompute` is `true`, then `store.res` is updated at the current value of 
        `x` before the objective function is computed. Note, that if the `store`
        is just initialized, `recompute` should be set to `true`, otherwise
        the returned value might not be correct.
    """
    function NLPModels.obj($(args_store...); recompute::Bool=true) where {T,S}
        # Only update residual at x if recompute is true, otherwise just compute 
        # objective function value with current value of r (residual)
        recompute && residual!(progData, preComp, store, x)
        increment!(progData, :neval_obj)
        return T(0.5 * dot(store.res, store.res))
    end

    @doc """
        jac_residual!(
            $(join(string.(args_store),",\n\t    "))
        ) where{T,S}
        
    Computes the Jacobian of the residual at `x`. This is just `prog.coef`, which is
        already maintained in `store.jac`. Only the number of Jacobian evaluations
        is incremented; no other calculations are performed.
    """
    function jac_residual!($(args_store...); recompute::Bool=true) where {T,S}
        increment!(progData, :neval_jac_residual)
        return nothing 
    end

    @doc """
        grad!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}
        
    Compute the gradient of the objective function at `x`, which is `A'(A*x-b)`.
        This calculation is performed by using precomputed values of `A'A` and 
        `A'b`. The value of `store.grad` is updated. 
    """
    function NLPModels.grad!($(args_store...); recompute::Bool=true) where {T,S}
        increment!(progData, :neval_grad)
        store.grad .= preComp.coef_t_coef * x - preComp.coef_t_cons
        return nothing
    end

   @doc """
        objgrad!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}
    
    Simultaneously computes the objective function and the gradient function.
        The objective function is computed as `0.5(x'A'Ax - 2x'A'b + b'b)`.
        The gradient function is computed as `A'Ax - A'b`. 
    """
    function NLPModels.objgrad!($(args_store...)) where {T,S}
        increment!(progData, :neval_obj)
        increment!(progData, :neval_grad)
        store.grad .= preComp.coef_t_coef * x - preComp.coef_t_cons

        #Store grad = A'A*x - A'b; 
        #We use this to compute (x'A'Ax - x'A'b) - (x'A'b) + b'b. 
        obj_value = T(0.5)*( dot(x, store.grad) - dot(x, preComp.coef_t_cons) +
            preComp.cons_t_cons)
        return obj_value
    end

    @doc """
        hess!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}

    Computes the Hessian of the objective function at `x`, which is `A'A`. This is
        already stored in `preComp.coef_t_coef` and `store.hess`. So only the 
        counter is updated. No other calculations are performed.
    """
    function hess!($(args_store...); recompute::Bool=true) where {T,S}
        increment!(progData, :neval_hess)
        return nothing
    end
end