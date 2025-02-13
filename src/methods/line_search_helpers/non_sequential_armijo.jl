# Date: 02/13/2025
# Author: Christian Varner
# Purpose: Implement the non-sequential armijo descent check

"""
TODO
"""
function non_sequential_armijo_condition(F_ψjk::S, reference_value::T, 
    norm_grad_θk::T, ρ::T, δk::T, α0k::T) where {T, S}

    return (F_ψjk < reference_value - ρ * δk * α0k * (norm_grad_θk ^ 2))
end