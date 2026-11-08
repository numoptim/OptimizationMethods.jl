# Shared definitions for the analysis (Phase 3). Environment: experiments/coap-revision/analysis
using CSV, DataFrames, Statistics, Printf

const TOLS = [1e-3, 1e-5, 1e-7, 1e-9]
const MAIN_SET = ["OURS", "ARM", "WATCH", "SPEC", "ADATR", "ASTR1", "LIPAPP"]
const ALL_METHODS = ["FIXED", "DIM", "BBGD", "LIPAPP", "NEST", "WEIGHT", "OURS", "ARM",
    "WATCH", "SPEC", "ADATR", "ASTR1"]
const METRICS = [:nobj, :ngrad, :cost, :time]
const METRIC_LABEL = Dict(:nobj => "#F", :ngrad => "#∇F", :cost => "weighted cost  #F·c + #∇F",
    :time => "wall time (s)")

tol_label(t) = @sprintf("%.0e", t) |> s -> replace(s, "e-0" => "e-")   # 1e-3, 1e-5, ...

"""
    load_merged(dir) -> (runs, hits, events)

Reads the merged CSVs written by `src/collect.jl`.
"""
function load_merged(dir)
    runs = CSV.read(joinpath(dir, "runs.csv"), DataFrame)
    hits = CSV.read(joinpath(dir, "hits.csv"), DataFrame)
    events = CSV.read(joinpath(dir, "ours_events.csv"), DataFrame)
    return runs, hits, events
end

"""
    build_table(runs, hits) -> DataFrame

One row per (run, tolerance) with the metrics at the first hit:
`nobj`, `ngrad`, `time` (wall seconds), and the weighted cost `cost = nobj*c + ngrad`
where `c = (mean time per F)/(mean time per ∇F)` is computed per run from the whole
run (`time_obj_s/n_obj` over `time_grad_s/n_grad`); runs with no F evaluations have
`c = 0`. Metrics are `NaN` when the tolerance was not hit. `reliable` is the paper's
definition (hit and F(x_hit) < F(x0)).
"""
function build_table(runs::DataFrame, hits::DataFrame)
    r = DataFrame(problem_id = runs.problem_id, start_id = runs.start_id,
        algorithm_id = runs.algorithm_id, method = runs.method,
        c = [(a > 0 && b > 0 && tg > 0) ? (to / a) / (tg / b) : 0.0
             for (a, b, to, tg) in zip(runs.n_obj, runs.n_grad, runs.time_obj_s, runs.time_grad_s)])
    T = innerjoin(hits, r, on = [:problem_id, :start_id, :algorithm_id])
    f(x) = [ismissing(v) ? NaN : Float64(v) for v in x]
    T.nobj = f(T.n_obj); T.ngrad = f(T.n_grad); T.time = f(T.time_total_s)
    T.iteration = f(T.iteration)
    T.cost = T.nobj .* T.c .+ T.ngrad
    T.reliable = Bool.(T.reliable)
    return select(T, :problem_id, :start_id, :algorithm_id, :method, :tol, :hit, :reliable,
        :iteration, :nobj, :ngrad, :cost, :time, :c)
end

"Normal-approximation 95% margin of error of the mean, 1.96 sd/√n (0 if n < 2)."
moe95(x) = length(x) < 2 ? 0.0 : 1.96 * std(x) / sqrt(length(x))
sd0(x) = length(x) < 2 ? 0.0 : std(x)

"""
    best_configs(T, tol; methods) -> DataFrame(method, algorithm_id, reliability, median_ngrad)

For each method the configuration with the highest reliability at `tol`; ties are
broken by the smaller median #∇F over the reliable runs, then by the smaller id.
"""
function best_configs(T::DataFrame, tol::Real; methods = ALL_METHODS)
    sub = T[T.tol .== tol, :]
    out = NamedTuple[]
    for g in groupby(sub, [:method, :algorithm_id])
        rel = mean(g.reliable)
        med = any(g.reliable) ? median(g.ngrad[g.reliable]) : Inf
        push!(out, (method = g.method[1], algorithm_id = g.algorithm_id[1],
            reliability = rel, median_ngrad = med))
    end
    df = DataFrame(out)
    best = NamedTuple[]
    for m in methods
        d = df[df.method .== m, :]
        nrow(d) == 0 && continue
        sort!(d, [order(:reliability, rev = true), :median_ngrad, :algorithm_id])
        push!(best, NamedTuple(d[1, :]))
    end
    return DataFrame(best)
end
