# Date: 2025/04/22
# Author: Christian Varner
# Purpose: Implementation of an BFGS update

"""
    update_bfgs!(H::Matrix{T}, r::Vector{T}, update::Matrix{T}, s::Vector{T},
        y::Vector{T}; damped_update::Bool = true)

Update the matrix `H` using the BFGS update formula or the damped BFGS
    update formula when `damped_update = true`. This function modifies `H`
    in-place for the new approximation, modifies `r` in-place to store
    the vector used in the BFGS update formula, and modifies `update` 
    in-place with the update matrix applied to `H`. 

!!! note
    If the denominator of the update formula is `0`, the matrix `H` is not
    updated at all.

# Reference(s)

[Nocedal and Wright, "Numerical Optimization". Springer. 2nd Edition. 
    Chapter 6 and 18.](@cite nocedal2006Numerical)

# Method

Let ``B_k \\in \\mathbb{R}^{n \\times n}`` be the ``k^{th}`` 
positive definite Hessian approximation, 
``\\theta_{k+1} \\in \\mathbb{R}^n`` and
``\\theta_k \\in \\mathbb{R}^n`` iterate of an optimization algorithm, 
``s_{k} = \\theta_{k+1} - \\theta_k``,
and ``y_{k} = \\nabla F(\\theta_{k+1}) - \\nabla F(\\theta_k)`` for a
``k + 1 \\in \\mathbb{N}``. Then, this function forms 
``B_{k+1} \\in \\mathbb{R}^{n \\times n}`` depending on the value of
`damped_update`.

## Non-damped Update (`damped_update == false`)

The non-damped update is the regular BFGS update.

```math
    B_{k+1} = B_k - \\frac{B_k s_k s_k^\\intercal B_k}{s_k^\\intercal B_k s_k} + 
        \\frac{y_k y_k^\\intercal}{s_k^\\intercal y_k}.
```

## Damped Update (`damped_update == true`)

Let

```math
    r_{k} = \\gamma_k y_k + (1 - \\gamma_k) B_k s_k, 
```

where ``\\gamma_k = 1`` when ``s_k^\\intercal y_k >= .2 s_k^\\intercal B_k s_k``
and ``\\gamma_k = .8 s_k^\\intercal B_k s_k / (s_k^\\intercal B_k s_k - s_k^\\intercal y_k)``
otherwise. Using this then, the update formula is

```math
    B_{k+1} = B_k - \\frac{B_k s_k s_k^\\intercal B_k}{s_k^\\intercal B_k s_k} + 
        \\frac{r_k r_k^\\intercal}{s_k^\\intercal r_k}.
```

# Arguments

- `H::Matrix{T}`, matrix that the (damped) BFGS update will be applied to.
- `r::Vector{T}`, buffer vector that store the vector used in the BFGS
    update formula
- `update::Matrix{T}`, buffer matrix that stores the update that is added
    to `H`.
- `s::Vector{T}`, vector used in the update; should correspond to the difference
    between two consecutive iterates when used in an optimization routine.
- `y::Vector{T};`, vector used in the update; should correspond to the difference
    between two consecutive gradient values when used in an optimization
    routine.

## (Optional) Keyword Argument

- `damped_update::Bool = true`, whether to use the damped BFGS update or
    the normal BFGS update

# Return

- A value of type `Bool`, which indicates whether the update was successfully
    applied or not.
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
