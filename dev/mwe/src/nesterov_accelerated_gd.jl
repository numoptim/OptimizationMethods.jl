"""
Nesterov acceleration preserving v0.1.2 (Li et al., arXiv:2306.01264, Algorithm 1).
With acceleration weight D, update B+=D, x=y-alpha*g(y), z-=alpha*D*g(y),
D=(1+sqrt(4B+1))/2 and y=x+D/(B+D+1/alpha)*(z-x).
The stopping norm is evaluated at x; the stored gradient is then evaluated at y.
"""
struct NesterovAcceleratedGD{T} <: AbstractMethod
    step_size::T
end
function NesterovAcceleratedGD(; step_size)
    a = FixedStepGD(; step_size).step_size
    return NesterovAcceleratedGD(a)
end
function allocate!(s, m::NesterovAcceleratedGD, p; kwargs...)
    any(k -> haskey(s,k), (:y,:z,:B,:D)) && throw(ArgumentError("key collision"))
    allocate_common!(s, p; kwargs...)
    s[:y] = copy(s[:x]); s[:z] = copy(s[:x])
    s[:B] = zero(eltype(s[:x])); s[:D] = one(eltype(s[:x]))
    return s
end
needed_grad(::NesterovAcceleratedGD) = 2
function step!(m::NesterovAcceleratedGD, p, s)
    a = m.step_size; D = s[:D]; B = s[:B] + D
    s[:x] .= s[:y] .- a .* s[:grad]
    s[:z] .-= a .* D .* s[:grad]
    D = oftype(D, (1 + sqrt(4B + 1)) / 2)
    s[:y] .= s[:x] .+ D / (B + D + 1/a) .* (s[:z] .- s[:x])
    s[:B] = B; s[:D] = D
    grad!(p; store=s, x=s[:x]); s[:grad_norm] = norm(s[:grad])
    grad!(p; store=s, x=s[:y])
    return nothing
end
