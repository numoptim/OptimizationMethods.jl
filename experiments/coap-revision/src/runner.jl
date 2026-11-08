# Runs one (problem, start, algorithm) with the timing layer and the budgets, and
# returns the rows for runs.csv / hits.csv / ours_events.csv (schemas: PROMPT §5).

using CSV, DataFrames, Random, LinearAlgebra, Printf, Dates, NLPModels, OptimizationMethods
include(joinpath(@__DIR__, "instances.jl"))

const TOLS = [1e-3, 1e-5, 1e-7, 1e-9]
const THRESHOLD = 1e-9
const ALLOWED_STATUS = ["converged_1e-9", "budget_grad", "budget_obj", "timeout",
    "nan_or_inf", "error", "stalled"]

base_counters(c::NLPModels.NLSCounters) = c.counters
base_counters(c) = c

"""
    make_algorithm(r, x0, threshold, max_iterations) -> (optData, solver_function)

Constructs the optimizer for row `r` of config/algorithms.csv (Appendix C of the
prompt plus the new methods).
"""
function make_algorithm(r, x0::Vector{Float64}, thr::Float64, maxit::Int)
    T = Float64
    m = r.method
    η = ismissing(r.eta) ? NaN : Float64(r.eta)
    if m == "FIXED"
        return FixedStepGD(T; x0 = x0, step_size = η, threshold = thr, max_iterations = maxit), fixed_step_gd
    elseif m == "DIM"
        return DiminishingStepGD(T; x0 = x0, step_size_function = OM.stepdown_100_step_size,
            step_size_scaling = η, threshold = thr, max_iterations = maxit), diminishing_step_gd
    elseif m == "BBGD"
        return BarzilaiBorweinGD(T; x0 = x0, init_stepsize = η, long_stepsize = Bool(r.long_stepsize),
            threshold = thr, max_iterations = maxit), barzilai_borwein_gd
    elseif m == "LIPAPP"
        return LipschitzApproxGD(T; x0 = x0, init_stepsize = η, threshold = thr,
            max_iterations = maxit), lipschitz_approximation_gd
    elseif m == "NEST"
        return NesterovAcceleratedGD(T; x0 = x0, step_size = η, threshold = thr,
            max_iterations = maxit), nesterov_accelerated_gd
    elseif m == "WEIGHT"
        return WeightedNormDampingGD(T; x0 = x0, init_norm_damping_factor = 1 / η,
            threshold = thr, max_iterations = maxit), weighted_norm_damping_gd
    elseif m == "OURS"
        return NonsequentialArmijoAdaptiveGD(T; x0 = x0, δ0 = 1.0, δ_upper = 1.0, ρ = 1e-4,
            threshold = thr, max_iterations = maxit), nonsequential_armijo_adaptive_gd
    elseif m == "ARM"
        return BacktrackingGD(T; x0 = x0, α = 1.0, δ = 0.5, ρ = 1e-4,
            line_search_max_iteration = 100, threshold = thr, max_iterations = maxit), backtracking_gd
    elseif m == "WATCH"
        return WatchdogFixedSimpleGD(T; x0 = x0, α = η, δ = 0.5, ρ = 1e-4,
            line_search_max_iterations = 100, window_size = 1, inner_loop_max_iterations = 100,
            threshold = thr, max_iterations = maxit), watchdog_fixed_simple_gd
    elseif m == "SPEC"
        return SafeBarzilaiBorweinNLSMaxValGD(T; x0 = x0, δ = 0.5, ρ = 1e-4, window_size = 10,
            line_search_max_iteration = 100, init_stepsize = η, long_stepsize = Bool(r.long_stepsize),
            α_lower = 1e-10, α_default = 1.0, threshold = thr, max_iterations = maxit),
            safe_barzilai_borwein_nls_maxval_gd
    elseif m == "ADATR"
        return AdaptiveTrustRegionGD(T; x0 = x0, init_scaling = 1 / η, α = Float64(r.adatr_alpha),
            b_min = 1e-4, threshold = thr, max_iterations = maxit), adaptive_trust_region_gd
    elseif m == "ASTR1"
        return ASTR1GD(T; x0 = x0, variant = Symbol(r.astr1_variant), θ = 1 / η,
            threshold = thr, max_iterations = maxit), astr1_gd
    else
        error("unknown method $m")
    end
end

"""
    uncounted(f, prob)

Runs `f()` and restores the objective/gradient counters of `prob` afterwards, so
the evaluations done for the analysis are never counted.
"""
function uncounted(f, prob)
    c = base_counters(prob.counters)
    nobj, ngrad = c.neval_obj, c.neval_grad
    try
        return f()
    finally
        c.neval_obj = nobj
        c.neval_grad = ngrad
    end
end

function safe_objective(prob, precomp, store, x)
    all(isfinite, x) || return NaN
    return try
        uncounted(prob) do
            Float64(OM.obj(prob, precomp, store, x))
        end
    catch
        NaN
    end
end

const HOSTNAME = gethostname()
const CPU_MODEL = try Sys.cpu_info()[1].model catch; "unknown" end

"""
    run_one(prob, alg, x0; G_max, F_max, timeout_s, ids...) -> NamedTuple

Runs algorithm row `alg` from `x0` on `prob`. Never throws: exceptions give
status `error` (message in `errmsg`). Returns `(run, hits, events, errmsg)`.
"""
function run_one(prob, alg, x0::Vector{Float64}; G_max::Int, F_max::Int, timeout_s::Float64,
        problem_row, start_id::Int, job_id)
    precomp, store = OM.initialize(prob)          # for uncounted evaluations
    F0 = safe_objective(prob, precomp, store, x0)
    gradnorm0 = uncounted(prob) do
        OM.grad!(prob, precomp, store, x0)
        norm(store.grad)
    end

    timed = TimedProblem(prob; tols = TOLS, max_grad = G_max, max_obj = F_max,
        max_time_ns = round(Int, timeout_s * 1e9))
    maxit = G_max + 10                           # never binds (>= 1 gradient/iteration)
    status = "error"
    errmsg = ""
    x = copy(x0)
    optData = nothing
    t_total_ns = 0
    finished = false
    try
        optData, solver = make_algorithm(alg, copy(x0), THRESHOLD, maxit)
        start!(timed)
        xr, st = run_solver!(solver, optData, timed)
        t_total_ns = Int(time_ns() - timed.t_start_ns)
        x = xr
        if st == :budget_grad
            status = "budget_grad"
        elseif st == :budget_obj
            status = "budget_obj"
        elseif st == :timeout
            status = "timeout"
        else
            finished = true
            if timed.hits[end] !== nothing
                status = "converged_1e-9"
            elseif !isfinite(timed.last_gradnorm) || !all(isfinite, x)
                status = "nan_or_inf"
            else
                status = "stalled"       # solver stopped by itself (failed line search / max_iterations)
            end
        end
    catch e
        errmsg = sprint(showerror, e)
        t_total_ns = Int(time_ns() - timed.t_start_ns)
        x = timed.last_iter >= 0 ? copy(timed.last_x) : copy(x0)
        status = "error"
    end
    c = base_counters(prob.counters)
    n_obj, n_grad = Int(c.neval_obj), Int(c.neval_grad)
    iterations = finished && optData !== nothing ? optData.stop_iteration : timed.last_iter
    F_final = safe_objective(prob, precomp, store, x)
    gradnorm_final = timed.last_iter >= 0 ? Float64(timed.last_gradnorm) : NaN

    run = (problem_id = problem_row.problem_id, n = problem_row.n, m = problem_row.m,
        V = problem_row.V, replicate = problem_row.replicate, start_id = start_id,
        algorithm_id = alg.algorithm_id, method = alg.method, eta = alg.eta,
        long_stepsize = alg.long_stepsize, adatr_alpha = alg.adatr_alpha,
        astr1_variant = alg.astr1_variant, status = status, iterations = iterations,
        n_obj = n_obj, n_grad = n_grad, time_total_s = t_total_ns / 1e9,
        time_obj_s = timed.time_obj_ns / 1e9, time_grad_s = timed.time_grad_ns / 1e9,
        F0 = F0, F_final = F_final, gradnorm0 = gradnorm0, gradnorm_final = gradnorm_final,
        julia_version = string(VERSION), hostname = HOSTNAME, cpu_model = CPU_MODEL,
        job_id = job_id)

    hits = NamedTuple[]
    for (i, tol) in enumerate(TOLS)
        h = timed.hits[i]
        if h === nothing
            push!(hits, (problem_id = problem_row.problem_id, start_id = start_id,
                algorithm_id = alg.algorithm_id, tol = tol, hit = false, iteration = missing,
                n_obj = missing, n_grad = missing, time_total_s = missing, time_obj_s = missing,
                time_grad_s = missing, gradnorm = missing, F_at_hit = missing, reliable = false))
        else
            Fh = safe_objective(prob, precomp, store, h.x)
            reliable = isfinite(Fh) && isfinite(F0) && Fh < F0
            push!(hits, (problem_id = problem_row.problem_id, start_id = start_id,
                algorithm_id = alg.algorithm_id, tol = tol, hit = true, iteration = h.iteration,
                n_obj = h.n_obj, n_grad = h.n_grad, time_total_s = h.time_total_ns / 1e9,
                time_obj_s = h.time_obj_ns / 1e9, time_grad_s = h.time_grad_ns / 1e9,
                gradnorm = Float64(h.gradnorm), F_at_hit = Fh, reliable = reliable))
        end
    end

    events = NamedTuple[]
    if alg.method == "OURS" && optData !== nothing
        for (k, j) in timed.inner_loops
            ev = optData.triggering_event_hist[k]
            # the acceptance flag is only set once the outer iteration k completed
            accepted = k <= timed.last_iter ? Bool(ev[5]) : missing
            push!(events, (problem_id = problem_row.problem_id, start_id = start_id,
                algorithm_id = alg.algorithm_id, k = k, j_k = j, ball = Bool(ev[1]),
                lower = Bool(ev[2]), upper = Bool(ev[3]), maxiter = Bool(ev[4]),
                accepted = accepted))
        end
    end
    return (run = run, hits = hits, events = events, errmsg = errmsg)
end

parse_ids(s) = isempty(strip(string(s))) ? Int[] : parse.(Int, split(string(s), ';'))
