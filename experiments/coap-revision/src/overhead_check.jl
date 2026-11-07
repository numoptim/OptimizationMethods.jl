# Timing-layer overhead: plain vs TimedProblem, n=90, m=1000, all four variance
# functions, FixedStepGD (gradient only) and BacktrackingGD (F and grad).
# Run: julia --project=. experiments/coap-revision/src/overhead_check.jl
using OptimizationMethods, Random, NLPModels, Statistics
const OM = OptimizationMethods
Random.seed!(1)
T = Float64
n, m = 90, 1000
probs = Dict(
    "V1" => OM.QLLogisticSin(T; nobs = m, nvar = n),
    "V2" => OM.QLLogisticMonomial(T; nobs = m, nvar = n, p = 2.25, c = 1.0),
    "V3" => OM.QLLogisticCenteredExp(T; nobs = m, nvar = n, p = 2.25, c = 1.0),
    "V4" => OM.QLLogisticCenteredLog(T; nobs = m, nvar = n, p = 2.25, c = 1.0, d = 1.0),
)
mkfixed(x0) = FixedStepGD(T; x0 = copy(x0), step_size = 1.0, threshold = 0.0, max_iterations = 300)
mkarm(x0) = BacktrackingGD(T; x0 = copy(x0), α = 1.0, δ = 0.5, ρ = 1e-4,
    line_search_max_iteration = 100, threshold = 0.0, max_iterations = 100)
function best_time(f; reps = 7)
    f()  # compile
    minimum(@elapsed(f()) for _ in 1:reps)
end
x0 = 20 .* rand(T, n) .- 10
for V in sort(collect(keys(probs)))
    p = probs[V]
    for (name, mk, run) in (("FIXED", mkfixed, fixed_step_gd), ("ARM", mkarm, backtracking_gd))
        tp = best_time(() -> (NLPModels.reset!(p.counters); run(mk(x0), p)))
        tt = best_time(() -> (tm = TimedProblem(p); start!(tm); run(mk(x0), tm)))
        println(rpad(V, 3), rpad(name, 6), "plain=", round(tp, digits = 4), "s timed=",
            round(tt, digits = 4), "s overhead=", round(100 * (tt - tp) / tp, digits = 2), "%")
    end
end
