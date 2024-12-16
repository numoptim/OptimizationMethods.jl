# Date: 12/16/2024
# Author: Christian Varner
# Purpose: Helper functions for diminishing_gd() to compute
# step sizes based on the iteration number

"""
    inverse_k_step_size(::Type{T}, k::Int64)

Return the inverse of `k`.

# Method

The step size sequence generated for ``k \\in \\mathbb{N}`` is
```math
    \\alpha_k = \\frac{1}{k},
```
when using this method.

# Arguments

- `T::Type`, data type that the computations are done in.
- `k::Int64`, index of the step size needed. 

# Returns

Returns a number of type `T`.
"""
function inverse_k_step_size(::Type{T}, k::Int64) where {T}
    return T(1/k)
end

"""
    inverse_log2k_step_size(::Type{T}, k::Int64) where {T}

# Method

The step size sequence generated when using this method is
```math
    \\alpha_k = \\frac{1}{\\lfloor \\log_2(k) + 1 \\rfloor}
```
for ``k \\in \\mathbb{N}``.

# Arguments

- `T::Type`, data type that the computation are done in.
- `k::Int64`, index of the step size needed.

# Returns

Returns a number of type `T`.
"""
function inverse_log2k_step_size(::Type{T}, k::Int64) where {T}
    return T(1 / (floor(log(2, k) + 1)))
end
