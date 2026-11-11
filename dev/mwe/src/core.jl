"""Immutable method configurations implement allocation, initialization and one step."""
abstract type AbstractMethod end
struct History
    keys::Tuple{Vararg{Symbol}}
    every::Int
    function History(; keys=[:x, :grad_norm], every=1)
        every > 0 || throw(ArgumentError("every must be positive"))
        ks = Tuple(keys)
        length(unique(ks)) == length(ks) || throw(ArgumentError("duplicate history keys"))
        any(k -> k in (:history, :history_spec, :iter), ks) && throw(ArgumentError("reserved history key"))
        new(ks, every)
    end
end
struct StoppingCriteria
    grad_norm::Float64
    max_iterations::Int
    max_grad_evals::Int
    max_obj_evals::Int
    function StoppingCriteria(; grad_norm=1e-6, max_iterations=1000,
        max_grad_evals=typemax(Int), max_obj_evals=typemax(Int))
        isfinite(grad_norm) && grad_norm >= 0 || throw(ArgumentError("invalid tolerance"))
        min(max_iterations, max_grad_evals, max_obj_evals) >= 0 || throw(ArgumentError("negative budget"))
        new(grad_norm, max_iterations, max_grad_evals, max_obj_evals)
    end
end
function allocate_common!(s, p::OptimizationProblem; x0, history=nothing)
    reserved = (:x, :iter, :grad_norm, :counters, :stop_reason, :history, :history_spec, :stop_spec)
    any(k -> haskey(s, k), reserved) && throw(ArgumentError("method keys already present; allocate a fresh store"))
    x0 isa Vector{<:AbstractFloat} || throw(ArgumentError("x0 must be a floating vector"))
    length(x0) == p.num_param || throw(DimensionMismatch("x0"))
    eltype(x0) == eltype(s[:grad]) || throw(ArgumentError("x0 and gradient buffer types must match"))
    all(isfinite, x0) || throw(ArgumentError("nonfinite x0"))
    s[:x] = copy(x0); s[:iter] = 0; s[:grad_norm] = Inf
    s[:stop_reason] = :none; s[:counters] = p.counters
    if history !== nothing
        history isa History || throw(ArgumentError("invalid history"))
        s[:history_spec] = history
        s[:history] = Dict{Symbol,Any}(k => Any[] for k in history.keys)
        s[:history][:iter] = Int[]
    end
    return s
end
count(s, k) = s[:counters][k].batch_equivalent
function check_stop(stop::StoppingCriteria, s)
    s[:grad_norm] <= stop.grad_norm && return :grad_norm
    s[:iter] >= stop.max_iterations && return :max_iterations
    count(s, :grad) >= stop.max_grad_evals && return :max_grad_evals
    count(s, :obj) >= stop.max_obj_evals && return :max_obj_evals
    return :none
end
"""Snapshot existing store values only; force records terminal states without duplicate iterations."""
function record!(s; force=false)
    haskey(s, :history_spec) || return nothing
    h = s[:history]; spec = s[:history_spec]; k = s[:iter]
    (force || k % spec.every == 0) || return nothing
    replace = !isempty(h[:iter]) && last(h[:iter]) == k
    for key in spec.keys
        value = deepcopy(s[key])
        replace ? (h[key][end] = value) : push!(h[key], value)
    end
    replace || push!(h[:iter], k)
    return nothing
end
needed_grad(::AbstractMethod) = 1
function optimize!(m::AbstractMethod, p, s; stop=StoppingCriteria())
    s[:iter] == 0 && s[:stop_reason] == :none || throw(ArgumentError("fresh store required"))
    s[:stop_spec] = stop
    if count(s, :grad) >= stop.max_grad_evals
        s[:stop_reason] = :max_grad_evals
        record!(s; force=true)
        return s[:x]
    end
    initialize!(m, p, s)
    record!(s; force=true)
    s[:stop_reason] = isfinite(s[:grad_norm]) ? check_stop(stop, s) : :nonfinite
    while s[:stop_reason] == :none
        if stop.max_grad_evals - count(s, :grad) < needed_grad(m)
            s[:stop_reason] = :max_grad_evals
            break
        end
        step!(m, p, s)
        s[:stop_reason] != :none && break
        s[:iter] += 1
        if !isfinite(s[:grad_norm]) || !all(isfinite, s[:x])
            s[:stop_reason] = :nonfinite
        else
            s[:stop_reason] = check_stop(stop, s)
        end
        record!(s)
    end
    record!(s; force=true)
    return s[:x]
end
function initialize!(::AbstractMethod, p, s)
    grad!(p; store=s, x=s[:x])
    s[:grad_norm] = norm(s[:grad])
    return nothing
end
