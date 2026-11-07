# Timing layer and tolerance first-hit recorder (COAP revision experiments).
#
# `TimedProblem` wraps any problem that has `initialize`, `obj`, `grad!`. It
# shares the NLPModels counters of the wrapped problem, times every objective
# and gradient call with `time_ns()`, and records the first time a solver's own
# termination test would fire at each tolerance in `tols`. Solvers call
# `record_iterate!(progData, iter, x, gra_norm)` right after computing the
# gradient norm that their termination test uses; for ordinary problems this is
# a no-op.

"""
    ToleranceHit{T}

First time the termination test of a solver would have fired at tolerance `tol`.

# Fields
- `tol::Float64`, the tolerance
- `iteration::Int`, iteration index at the hit
- `n_obj::Int`, `n_grad::Int`, cumulative NLPModels counters at the hit
- `time_total_ns::Int`, `time_obj_ns::Int`, `time_grad_ns::Int`, cumulative
    wall time, time inside objective calls and time inside gradient calls
- `gradnorm::T`, gradient norm at the hit
- `x::Vector{T}`, copy of the iterate
"""
struct ToleranceHit{T}
    tol::Float64
    iteration::Int
    n_obj::Int
    n_grad::Int
    time_total_ns::Int
    time_obj_ns::Int
    time_grad_ns::Int
    gradnorm::T
    x::Vector{T}
end

"""
    TimedProblem{T,S,P} <: AbstractNLPModel{T,S}

Wrapper around the problem `inner` that times objective/gradient calls and
records tolerance first-hits. Construct with `TimedProblem(inner)` or
`TimedProblem(inner; tols = [1e-3, 1e-5, 1e-7, 1e-9])`; call
`start!(timed)` immediately before running a solver.

# Fields
- `meta`, `counters`: shared with `inner` (so counts are NLPModels counts)
- `inner::P`, the wrapped problem
- `tols::Vector{Float64}`, tolerances, in the order they are reported
- `hits::Vector{Union{Nothing,ToleranceHit{T}}}`, one slot per tolerance
- `time_obj_ns::Int`, `time_grad_ns::Int`, accumulated call times
- `t_start_ns::UInt64`, time of the last `start!`
"""
mutable struct TimedProblem{T,S,P<:AbstractNLPModel{T,S},C} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::C
    inner::P
    tols::Vector{Float64}
    hits::Vector{Union{Nothing,ToleranceHit{T}}}
    time_obj_ns::Int
    time_grad_ns::Int
    t_start_ns::UInt64
end

function TimedProblem(
    inner::P;
    tols::Vector{Float64} = [1e-3, 1e-5, 1e-7, 1e-9],
) where {T,S,P<:AbstractNLPModel{T,S}}
    return TimedProblem{T,S,P,typeof(inner.counters)}(
        inner.meta, inner.counters, inner, tols,
        Union{Nothing,ToleranceHit{T}}[nothing for _ in tols],
        0, 0, time_ns(),
    )
end

"""
    start!(timed::TimedProblem)

Reset hits, call timers and the wrapped problem's counters, and start the
total-time clock.
"""
function start!(timed::TimedProblem{T}) where {T}
    NLPModels.reset!(timed.counters)
    fill!(timed.hits, nothing)
    timed.time_obj_ns = 0
    timed.time_grad_ns = 0
    timed.t_start_ns = time_ns()
    return timed
end

_base_counters(c::Counters) = c
_base_counters(c::NLPModels.NLSCounters) = c.counters

# interface forwarding
initialize(timed::TimedProblem) = initialize(timed.inner)

function NLPModels.obj(timed::TimedProblem, precomp, store, x; kwargs...)
    t0 = time_ns()
    val = NLPModels.obj(timed.inner, precomp, store, x; kwargs...)
    timed.time_obj_ns += Int(time_ns() - t0)
    return val
end

function NLPModels.grad!(timed::TimedProblem, precomp, store, x; kwargs...)
    t0 = time_ns()
    val = NLPModels.grad!(timed.inner, precomp, store, x; kwargs...)
    timed.time_grad_ns += Int(time_ns() - t0)
    return val
end

function NLPModels.objgrad!(timed::TimedProblem, precomp, store, x; kwargs...)
    t0 = time_ns()
    val = NLPModels.objgrad!(timed.inner, precomp, store, x; kwargs...)
    # attributed to the gradient clock (not used by the experiment solvers)
    timed.time_grad_ns += Int(time_ns() - t0)
    return val
end

"""
    record_iterate!(progData, iter, x, gra_norm)

Hook called by solvers right after the gradient norm used in their termination
test is computed. No-op unless `progData` is a `TimedProblem`, in which case it
stores a [`ToleranceHit`](@ref) for each tolerance `ε` not yet hit for which the
solver's test `gra_norm > ε` is false (so NaN counts as a hit, exactly as the
solver would stop).
"""
record_iterate!(progData, iter, x, gra_norm) = nothing

function record_iterate!(timed::TimedProblem{T}, iter, x, gra_norm) where {T}
    t = time_ns()
    for (i, ε) in enumerate(timed.tols)
        if timed.hits[i] === nothing && !(gra_norm > ε)
            timed.hits[i] = ToleranceHit{T}(
                ε, iter,
                Int(_base_counters(timed.counters).neval_obj),
                Int(_base_counters(timed.counters).neval_grad),
                Int(t - timed.t_start_ns), timed.time_obj_ns, timed.time_grad_ns,
                T(gra_norm), copy(x),
            )
        end
    end
    return nothing
end
