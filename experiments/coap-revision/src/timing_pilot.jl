# Times 1,000 F and 1,000 gradient evaluations for each (n, m, V) (replicate 1 data,
# evaluation points cycling through the 10 recorded starting points).
# Output: results/pilot/eval_times.csv   (t_F_s, t_grad_s = seconds per evaluation)
# Run: julia --project=. src/timing_pilot.jl
include(joinpath(@__DIR__, "runner.jl"))

problems = CSV.read(joinpath(@__DIR__, "..", "config", "problems.csv"), DataFrame)
starts = CSV.read(joinpath(@__DIR__, "..", "config", "starts.csv"), DataFrame)
rows = NamedTuple[]
NEVAL = 1000
for V in VARIANCE_LABELS, n in N_VALUES, m in M_VALUES
    pr = problems[(problems.V .== V) .& (problems.n .== n) .& (problems.m .== m) .& (problems.replicate .== 1), :][1, :]
    prob = build_problem(pr.V, pr.n, pr.m, pr.seed)
    precomp, store = OM.initialize(prob)
    xs = [build_start(n, starts.seed[(starts.problem_id .== pr.problem_id) .& (starts.start_id .== s)][1]) for s in 1:N_STARTS]
    OM.obj(prob, precomp, store, xs[1]); OM.grad!(prob, precomp, store, xs[1])   # compile
    tF = @elapsed for i in 1:NEVAL
        OM.obj(prob, precomp, store, xs[mod1(i, N_STARTS)])
    end
    tG = @elapsed for i in 1:NEVAL
        OM.grad!(prob, precomp, store, xs[mod1(i, N_STARTS)])
    end
    push!(rows, (V = V, n = n, m = m, problem_id = pr.problem_id, n_eval = NEVAL,
        t_F_s = tF / NEVAL, t_grad_s = tG / NEVAL, ratio_c = tF / tG))
    @printf("%s n=%2d m=%4d  t_F=%.3e s  t_grad=%.3e s  c=%.1f\n", V, n, m, tF / NEVAL, tG / NEVAL, tF / tG)
    flush(stdout)
end
mkpath(joinpath(@__DIR__, "..", "results", "pilot"))
CSV.write(joinpath(@__DIR__, "..", "results", "pilot", "eval_times.csv"), DataFrame(rows))
