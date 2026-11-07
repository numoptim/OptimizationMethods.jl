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
- `max_grad::Int`, `max_obj::Int`, `max_time_ns::Int`, evaluation budgets and
    wall-clock safety cap; a call that would start with `#∇F ≥ max_grad`
    (resp. `#F ≥ max_obj`, or elapsed time ≥ `max_time_ns`) throws
    [`BudgetExhausted`](@ref) instead. Use [`run_solver!`](@ref) to catch it.
- `inner_loops::Vector{Tuple{Int,Int}}`, `(k, j_k)` for every completed inner
    loop of `nonsequential_armijo_adaptive_gd` (see [`record_inner_loop!`](@ref))
- `last_x`, `last_iter`, `last_gradnorm`, the last iterate reported through
    `record_iterate!` (the last completed iteration of the solver)
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
    max_grad::Int
    max_obj::Int
    max_time_ns::Int
    last_x::Vector{T}
    last_iter::Int
    last_gradnorm::T
    inner_loops::Vector{Tuple{Int,Int}}
end

function TimedProblem(
    inner::P;
    tols::Vector{Float64} = [1e-3, 1e-5, 1e-7, 1e-9],
    max_grad::Int = typemax(Int),
    max_obj::Int = typemax(Int),
    max_time_ns::Int = typemax(Int),
) where {T,S,P<:AbstractNLPModel{T,S}}
    return TimedProblem{T,S,P,typeof(inner.counters)}(
        inner.meta, inner.counters, inner, tols,
        Union{Nothing,ToleranceHit{T}}[nothing for _ in tols],
        0, 0, time_ns(), max_grad, max_obj, max_time_ns,
        fill(T(NaN), length(inner.meta.x0)), -1, T(NaN),
        Tuple{Int,Int}[],
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
    fill!(timed.last_x, T(NaN))
    timed.last_iter = -1
    empty!(timed.inner_loops)
    return timed
end

_base_counters(c::Counters) = c
_base_counters(c::NLPModels.NLSCounters) = c.counters

"""
    BudgetExhausted(which::Symbol) <: Exception

Thrown by a [`TimedProblem`](@ref) before an evaluation that would exceed a
budget. `which` is `:grad`, `:obj` or `:time`.
"""
struct BudgetExhausted <: Exception
    which::Symbol
end

function _check_budget(timed::TimedProblem; obj::Bool, grad::Bool)
    c = _base_counters(timed.counters)
    grad && c.neval_grad >= timed.max_grad && throw(BudgetExhausted(:grad))
    obj && c.neval_obj >= timed.max_obj && throw(BudgetExhausted(:obj))
    if timed.max_time_ns != typemax(Int) &&
        Int(time_ns() - timed.t_start_ns) >= timed.max_time_ns
        throw(BudgetExhausted(:time))
    end
    return nothing
end

# interface forwarding
initialize(timed::TimedProblem) = initialize(timed.inner)

function NLPModels.obj(timed::TimedProblem, precomp, store, x; kwargs...)
    _check_budget(timed; obj = true, grad = false)
    t0 = time_ns()
    val = NLPModels.obj(timed.inner, precomp, store, x; kwargs...)
    timed.time_obj_ns += Int(time_ns() - t0)
    return val
end

function NLPModels.grad!(timed::TimedProblem, precomp, store, x; kwargs...)
    _check_budget(timed; obj = false, grad = true)
    t0 = time_ns()
    val = NLPModels.grad!(timed.inner, precomp, store, x; kwargs...)
    timed.time_grad_ns += Int(time_ns() - t0)
    return val
end

function NLPModels.objgrad!(timed::TimedProblem, precomp, store, x; kwargs...)
    _check_budget(timed; obj = true, grad = true)
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

"""
    record_inner_loop!(progData, k, j)

Hook called by `nonsequential_armijo_adaptive_gd` after each inner loop with the
outer index `k` and the inner-loop length `j_k`. No-op unless `progData` is a
`TimedProblem`, which appends `(k, j)` to `inner_loops`.
"""
record_inner_loop!(progData, k, j) = nothing
record_inner_loop!(timed::TimedProblem, k, j) = (push!(timed.inner_loops, (Int(k), Int(j))); nothing)

function record_iterate!(timed::TimedProblem{T}, iter, x, gra_norm) where {T}
    t = time_ns()
    copyto!(timed.last_x, x)
    timed.last_iter = iter
    timed.last_gradnorm = gra_norm
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

"""
    run_solver!(solver, optData, timed::TimedProblem)

Calls `solver(optData, timed)` (after nothing else; call [`start!`](@ref) first)
and stops cleanly if a budget of `timed` is exhausted, including in the middle
of a line search or an inner loop. Returns `(x, status)` where `status` is
`:finished` (the solver stopped by itself: threshold, `max_iterations` or a
failed line search), `:budget_grad`, `:budget_obj` or `:timeout`. After budget
exhaustion, `x` is a copy of the last completed iterate (iteration
`timed.last_iter`; if `last_iter == -1`, nothing completed and `x` is all NaN), and the counters hold exactly what was evaluated.
"""
function run_solver!(solver, optData, timed::TimedProblem)
    try
        x = solver(optData, timed)
        return x, :finished
    catch e
        e isa BudgetExhausted || rethrow()
        status = e.which == :grad ? :budget_grad :
                 e.which == :obj ? :budget_obj : :timeout
        return copy(timed.last_x), status
    end
end
