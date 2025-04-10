# Date: 2025/04/10
# Author: Christian Varner
# Purpose: Given a symmetric matrix H, return a matrix H + λI such 
# that H + λI is positive semi-definite. The algorithm that is implemented
# below is Algorithm 3.3 from Nocedal and Wright, "Numerical Optimization".

"""
"""
function add_identity(
    A::Matrix{T},
    λ::T
) where {T}
    m = size(A)[1]
    for i in 1:m
        A[i, i] += λ
    end
end

"""
"""
function add_identity_until_psd!(
    res::Matrix{T};
    λ::T = T(0),
    β::T = T(1e-3),
    max_iterations::Int64 = 10
) where {T}

    iter = 0
    add_identity(res, λ)                                                
    while iter < max_iterations
        
        iter += 1
        
        C = cholesky(Hermitian(res); check = false)
        if issuccess(C)
            return C, λ, true
        else
            add_identity(res, -λ)
            λ = max(10 * λ, β)
            add_identity(res, λ)
        end
    end

    return nothing, λ, false
end 