# Date: 2025/04/10
# Author: Christian Varner
# Purpose: Given a symmetric matrix H, return a matrix H + λI such 
# that H + λI is positive semi-definite. The algorithm that is implemented
# below is Algorithm 3.3 from Nocedal and Wright, "Numerical Optimization".

"""
    add_identity(A::Matrix{T}, λ::T) where {T}

Add the scaler `λ` only to the diagonal of the matrix `A`, modifying `A` 
in place.

# Arguments

- `A::Matrix{T}`, matrix of values for which the diagonal will be editted.
- `λ::T`, constant that will be added to the matrix `A`
"""
function add_identity(
    A::Matrix{T},
    λ::T
) where {T}
    m = size(A)[1]
    for i in 1:m
        A[i, i] += λ
    end

    return nothing
end

"""
    add_identity_until_psd!(res::Matrix{T}; λ::T = T(0), β::T = T(1e-3),
        max_iterations::Int64 = 10) where {T}

Function that takes a matrix `res`, and tries to find a scaler `λ` such that
`res + λI` is positive definite. The matrix `res` is editted in-place, and
if a scaler is found, the upper triangular portion will be overwritten with
the cholesky factorization.

# Reference(s)

[Nocedal and Wright. "Numerical Optimization". Edition 2, Springer, 2006, 
    Page 51.](@cite nocedal2006Numerical)

# Method

The method we implement is Algorithm 3.3 on Page 51 in the above reference.

Given a symmetric matrix ``A \\in\\mathbb{R}^{n\\times n}``, we seek to
find a scalar, ``\\lambda \\in \\mathbb{R}_{\\geq 0}``, 
such that ``A + \\lambda I`` is positive definite. To accomplish this,
let ``\\lambda_0`` be some initial value and ``\\beta \\in \\mathbb{R}_{> 0}``.

The algorithm proceeds for each ``k-1 \\in \\mathbb{N}`` as follows:

1. Try a cholesky factorization of ``A + \\lambda_k I``, if this is successful
    then return the cholesky factorization.
2. If the cholesky factorization is not successful, 
    ``\\lambda_{k+1} = \\max(10 * \\lambda_k, \\beta)``, `k += 1`, and return
    to step 1.

While this is guaranteed to eventually terminate, to restrict the computational
cost of the method, there is an optional keyword argument `max_iterations` that
caps the number of times a cholesky factorization is attempted. By default it is
set to the value of `10`.

# Arguments

- `res::Matrix{T}`, matrix that will be made positive definite. Should be 
    symmetric. 

!!! warning
    The method wraps the matrix `res` with `Hermitian` to avoid numerical
    round-off errors when computing the hessian of a problem
    (see the documentation for `cholesky` in `LinearAlgebra`). Therefore,
    if `res` was not symmetric, the method might still return a successful
    factorization, which might be incorrect for the starting matrix.

## Optional Keyword Arguments

- `λ::T = T(0)`, the initial value for which a cholesky factorization of the 
    modified matrix is attempted for.
- `β::T = T(1e-3)`, value used to start the search procedure if `λ == 0` and
    this value failed (i.e., `res` is not positive definite).
- `max_iterations::Int64 = 10`, number of cholesky factorization to attempt.

# Returns

Return two values. First value is the value of `λ` for which the function
terminated on. The second value is a boolean flag indicating whether a factorization
was successfully found.
"""
function add_identity_until_pd!(
    res::Matrix{T};
    λ::T = T(0),
    β::T = T(1e-3),
    max_iterations::Int64 = 10
) where {T}

    iter = 0
    add_identity(res, λ + β)                                                
    while iter < max_iterations
        
        iter += 1
    
        if issuccess(cholesky!(Hermitian(res); check = false))
            return λ, true
        else
            add_identity(res, -λ)
            λ = max(10 * λ, β)
            add_identity(res, λ)
        end
    end

    return λ, false
end 