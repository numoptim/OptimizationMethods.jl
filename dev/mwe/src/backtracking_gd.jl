"""Gradient descent with Armijo sufficient decrease and geometrically reduced trial steps."""
struct BacktrackingGD{T} <: AbstractMethod
    step_size::T
    delta::T
    rho::T
    max_ls_iterations::Int
end
function BacktrackingGD(; step_size=1.0, delta=0.5, rho=1e-4, max_ls_iterations=100)
    a, d, r = promote(float(step_size), float(delta), float(rho))
    isfinite(a) && a > 0 && 0 < d < 1 && 0 < r < 1 && max_ls_iterations >= 0 || throw(ArgumentError("invalid line search configuration"))
    return BacktrackingGD(a, d, r, max_ls_iterations)
end
function allocate!(s, m::BacktrackingGD, p; kwargs...)
    any(k -> haskey(s,k), (:x_prev,:trial,:obj_x,:step_size,:ls_iterations)) && throw(ArgumentError("key collision"))
    allocate_common!(s, p; kwargs...)
    s[:x_prev] = copy(s[:x]); s[:trial] = similar(s[:x])
    s[:obj_x] = NaN; s[:step_size] = m.step_size; s[:ls_iterations] = 0
    return s
end
function step!(m::BacktrackingGD, p, s)
    backtracking!(m, p, s) || return nothing
    grad!(p; store=s, x=s[:x])
    s[:grad_norm] = norm(s[:grad])
    return nothing
end
