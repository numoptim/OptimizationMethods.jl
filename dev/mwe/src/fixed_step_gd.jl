"""
    FixedStepGD(; step_size)

Apply ``x_{k+1}=x_k-\\alpha\\nabla f(x_k)``, where alpha is a positive step size.
"""
struct FixedStepGD{T<:AbstractFloat} <: AbstractMethod
    step_size::T
    function FixedStepGD(; step_size)
        isfinite(step_size) && step_size > 0 || throw(ArgumentError("invalid step size"))
        new{typeof(float(step_size))}(float(step_size))
    end
end
allocate!(s, m::FixedStepGD, p; kwargs...) = allocate_common!(s, p; kwargs...)
update!(x, g, a) = (x .-= a .* g; nothing)
function step!(m::FixedStepGD, p, s)
    update!(s[:x], s[:grad], m.step_size)
    grad!(p; store=s, x=s[:x])
    s[:grad_norm] = norm(s[:grad])
    return nothing
end
