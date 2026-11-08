# Tables (Phase 3.2).
include(joinpath(@__DIR__, "common.jl"))

const METRIC_SUFFIX = Dict(:nobj => "nobj", :ngrad => "ngrad", :cost => "cost", :time => "time")

"""
    reliability_table(T, bc, tol, metric) -> DataFrame

Appendix-D format, one row per method (best configuration `bc` at `tol`):
`algorithm.name, param, total.x, avg.reliability, sd.reliability, moe_plus.x,
moe_minus.x, total.y, avg.<metric>, sd.<metric>, moe_plus.y, moe_minus.y`.
`param` is the algorithm_id; `total.x` the number of problem-start pairs;
`total.y` the number of reliable runs, over which the metric is averaged;
margins are the 95% normal-approximation margins 1.96 sd/√n.
"""
function reliability_table(T::DataFrame, bc::DataFrame, tol::Real, metric::Symbol)
    mname = METRIC_SUFFIX[metric]
    cols = Dict{String,Vector}(k => [] for k in ["algorithm.name", "param", "total.x",
        "avg.reliability", "sd.reliability", "moe_plus.x", "moe_minus.x", "total.y",
        "avg.$mname", "sd.$mname", "moe_plus.y", "moe_minus.y"])
    for r in eachrow(bc)
        d = T[(T.tol .== tol) .& (T.algorithm_id .== r.algorithm_id), :]
        rel = Float64.(d.reliable)
        vals = d[d.reliable, metric]
        μ = mean(rel); m1 = moe95(rel)
        μy = isempty(vals) ? NaN : mean(vals); m2 = moe95(vals)
        for (k, v) in ("algorithm.name" => r.method, "param" => r.algorithm_id, "total.x" => nrow(d),
                "avg.reliability" => μ, "sd.reliability" => sd0(rel), "moe_plus.x" => μ + m1,
                "moe_minus.x" => μ - m1, "total.y" => length(vals), "avg.$mname" => μy,
                "sd.$mname" => sd0(vals), "moe_plus.y" => μy + m2, "moe_minus.y" => μy - m2)
            push!(cols[k], v)
        end
    end
    order = ["algorithm.name", "param", "total.x", "avg.reliability", "sd.reliability", "moe_plus.x",
        "moe_minus.x", "total.y", "avg.$mname", "sd.$mname", "moe_plus.y", "moe_minus.y"]
    return DataFrame([k => identity.(cols[k]) for k in order])
end

"Reliability of every one of the 100 algorithms at every tolerance (supplement table)."
function all_algorithms_table(T::DataFrame, algs::DataFrame)
    rows = NamedTuple[]
    for a in eachrow(algs)
        row = Dict{Symbol,Any}(:algorithm_id => a.algorithm_id, :method => a.method,
            :eta => a.eta, :long_stepsize => a.long_stepsize, :adatr_alpha => a.adatr_alpha,
            :astr1_variant => a.astr1_variant)
        for tol in TOLS
            d = T[(T.tol .== tol) .& (T.algorithm_id .== a.algorithm_id), :]
            rel = Float64.(d.reliable)
            row[Symbol("reliability_" * tol_label(tol))] = isempty(rel) ? NaN : mean(rel)
            row[Symbol("moe_" * tol_label(tol))] = moe95(rel)
        end
        push!(rows, NamedTuple(row))
    end
    return DataFrame(rows)
end

"""
    success_summary(T, bc_by_tol) -> DataFrame

Mean and median #F, #∇F, wall time and weighted cost over the reliable runs of the
best configuration of every method, per tolerance.
"""
function success_summary(T::DataFrame, bc_by_tol::Dict{Float64,DataFrame})
    rows = NamedTuple[]
    for tol in TOLS
        for r in eachrow(bc_by_tol[tol])
            d = T[(T.tol .== tol) .& (T.algorithm_id .== r.algorithm_id) .& T.reliable, :]
            f(col, g) = isempty(d) ? NaN : g(d[!, col])
            push!(rows, (tol = tol, method = r.method, algorithm_id = r.algorithm_id,
                reliability = r.reliability, n_success = nrow(d),
                mean_nobj = f(:nobj, mean), median_nobj = f(:nobj, median),
                mean_ngrad = f(:ngrad, mean), median_ngrad = f(:ngrad, median),
                mean_time_s = f(:time, mean), median_time_s = f(:time, median),
                mean_cost = f(:cost, mean), median_cost = f(:cost, median)))
        end
    end
    return DataFrame(rows)
end

"""
    events_table(events, T, tol) -> DataFrame

OURS triggering events (paper Table 4 format `Events, Average.Proportion.on.Inner.Loops,
Margin.of.Error`, in percent) restricted, for each run that reached `tol`, to the
inner loops of the outer iterations `k <= iteration of the first hit`. The proportion
of a flag is computed per run and averaged over runs (margin: 95%, 1.96 sd/√runs).
The flags are not mutually exclusive. The last row is the rejection rate: the
proportion of those outer iterations whose acceptance flag is false (loops whose
acceptance is unknown, because the budget ended mid-iteration, are excluded).
Extra columns: `runs` (OURS runs that reached `tol` and have >= 1 loop) and `loops`.
"""
function events_table(events::DataFrame, T::DataFrame, tol::Real)
    hh = hits_iteration(T, tol)
    flags = [:ball, :lower, :upper, :maxiter]
    labels = ["Bounding Ball Event", "Lower Gradient Interval", "Upper Gradient Interval", "Max Iteration"]
    props = Dict(f => Float64[] for f in flags)
    rej = Float64[]
    nloops = 0
    for g in groupby(events, [:problem_id, :start_id, :algorithm_id])
        key = (g.problem_id[1], g.start_id[1], g.algorithm_id[1])
        haskey(hh, key) || continue
        d = g[g.k .<= hh[key], :]
        nrow(d) == 0 && continue
        nloops += nrow(d)
        for f in flags
            push!(props[f], 100 * mean(Bool.(d[!, f])))
        end
        acc = collect(skipmissing(d.accepted))
        isempty(acc) || push!(rej, 100 * mean(.!Bool.(acc)))
    end
    nruns = length(props[:ball])
    rows = [(Events = labels[i], var"Average.Proportion.on.Inner.Loops" = nruns == 0 ? NaN : mean(props[f]),
             var"Margin.of.Error" = moe95(props[f]), runs = nruns, loops = nloops)
            for (i, f) in enumerate(flags)]
    push!(rows, (Events = "Rejected outer iterations", var"Average.Proportion.on.Inner.Loops" = isempty(rej) ? NaN : mean(rej),
        var"Margin.of.Error" = moe95(rej), runs = length(rej), loops = nloops))
    return DataFrame(rows)
end

"Dict (problem, start, algorithm) -> iteration of the first hit at `tol` (rows with a hit)."
function hits_iteration(T::DataFrame, tol::Real)
    d = T[(T.tol .== tol) .& T.hit, :]
    return Dict(zip(zip(d.problem_id, d.start_id, d.algorithm_id), Int.(d.iteration)))
end
