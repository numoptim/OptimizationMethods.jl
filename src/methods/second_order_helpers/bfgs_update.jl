# Date: 2025/04/22
# Author: Christian Varner
# Purpose: Implementation of an BFGS update

"""
    update_bfgs!(H::Matrix{T}, r::Vector{T}, update::Matrix{T}, s::Vector{T},
        y::Vector{T}; damped_update::Bool = false)
"""
function update_bfgs!(
    H::Matrix{T},
    r::Vector{T},
    update::Matrix{T},
    s::Vector{T},
    y::Vector{T}; 
    damped_update::Bool = true) where {T}

    # check to make sure calculation will be okay
    Hs = H * s
    sHs = dot(s, Hs) 
    if sHs == 0.0
        return false
    end

    # get the damped BFGS update
    sy = dot(s, y)
    if (!damped_update) || (dot(s, y) >= .2 * sHs)
        r .= y
    else
        θ = (.8 * sHs) / (sHs - sy)
        r .= θ .* y .+ (1 - θ) .* Hs
    end

    # check to make sure update will be okay
    sr = dot(s, r)
    if sr == 0.0
        return false
    end

    # update the hessian
    update .= -(Hs*transpose(Hs) ./ sHs)
    update .+= (r*transpose(r) ./ dot(s, r))  
    H .+= update

    return true
end
