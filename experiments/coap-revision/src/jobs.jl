# Builds config/jobs.csv (full experiment) or config/pilot_jobs.csv (--pilot) from
# results/pilot/eval_times.csv and the sizing rules of PROMPT §3.
#   julia --project=. src/jobs.jl            # full:   G_max = F_max = 200000
#   julia --project=. src/jobs.jl --pilot    # Gate-2 local pilot: 3 problems x 2 starts, small budgets
#
# A job = one problem x one "kind" (gradient-only algorithms, or algorithms that
# evaluate F) x a tile of (algorithms x starts). Worst-case run time is
#   T_run = F_max t_F + G_max t_grad   (kind obj),   G_max t_grad   (kind grad),
# and a job is sized so that sum of T_run <= TARGET_WORST_H hours (CHTC holds jobs
# after 72 h; we target <= 24 h with margin). Each run gets a wall-clock safety cap
# of 2 T_run.
using CSV, DataFrames, Printf, Statistics

const PILOT = "--pilot" in ARGS
const ROOT = joinpath(@__DIR__, "..")
const G_MAX = PILOT ? 1000 : 200_000
const F_MAX = PILOT ? 1000 : 200_000
const TARGET_WORST_H = 20.0           # worst-case job time target (limit is 24 h)
const MIN_TIMEOUT_S = 30.0            # floor for the per-run safety cap (JIT, noise)

problems = CSV.read(joinpath(ROOT, "config", "problems.csv"), DataFrame)
algs = CSV.read(joinpath(ROOT, "config", "algorithms.csv"), DataFrame)
times = CSV.read(joinpath(ROOT, "results", "pilot", "eval_times.csv"), DataFrame)

if PILOT
    pick(V, n, m) = problems.problem_id[(problems.V .== V) .& (problems.n .== n) .& (problems.m .== m) .& (problems.replicate .== 1)][1]
    problem_ids = [pick("V1", 10, 100), pick("V3", 50, 1000), pick("V2", 90, 1000)]   # smallest, middle, largest
    start_ids = [1, 2]
else
    problem_ids = problems.problem_id
    start_ids = collect(1:10)
end

grad_algs = algs.algorithm_id[.!algs.uses_obj]
obj_algs = algs.algorithm_id[algs.uses_obj]

rows = NamedTuple[]
job_id = 0
for p in problem_ids
    pr = problems[problems.problem_id .== p, :][1, :]
    tm = times[(times.V .== pr.V) .& (times.n .== pr.n) .& (times.m .== pr.m), :][1, :]
    for (kind, ids) in (("grad", grad_algs), ("obj", obj_algs))
        t_run = kind == "grad" ? G_MAX * tm.t_grad_s : F_MAX * tm.t_F_s + G_MAX * tm.t_grad_s
        timeout = max(MIN_TIMEOUT_S, 2 * t_run)
        r_max = max(1, floor(Int, TARGET_WORST_H * 3600 / t_run))      # runs per job
        if PILOT
            r_max = typemax(Int)                                         # one job per (problem, kind)
        end
        a_chunk = min(length(ids), r_max)
        s_chunk = min(length(start_ids), max(1, r_max ÷ a_chunk))
        for ai in 1:a_chunk:length(ids), si in 1:s_chunk:length(start_ids)
            a_sub = ids[ai:min(ai + a_chunk - 1, end)]
            s_sub = start_ids[si:min(si + s_chunk - 1, end)]
            global job_id += 1
            push!(rows, (job_id = job_id, problem_id = p, kind = kind,
                algorithm_ids = join(a_sub, ";"), start_ids = join(s_sub, ";"),
                G_max = G_MAX, F_max = F_MAX,
                timeout_grad_s = kind == "grad" ? timeout : 0.0,
                timeout_obj_s = kind == "obj" ? timeout : 0.0,
                n_runs = length(a_sub) * length(s_sub),
                est_worst_h = length(a_sub) * length(s_sub) * t_run / 3600))
        end
    end
end
jobs = DataFrame(rows)
out = joinpath(ROOT, "config", PILOT ? "pilot_jobs.csv" : "jobs.csv")
CSV.write(out, jobs)

total_runs = sum(jobs.n_runs)
println("wrote $out")
@printf("jobs: %d   runs: %d   (expected %d)\n", nrow(jobs), total_runs,
    length(problem_ids) * length(start_ids) * nrow(algs))
@printf("worst-case job time (h): max %.2f  median %.2f  mean %.2f\n",
    maximum(jobs.est_worst_h), median(jobs.est_worst_h), mean(jobs.est_worst_h))
@printf("worst-case total core-hours: %.0f   (obj kind: %.0f, grad kind: %.0f)\n",
    sum(jobs.est_worst_h), sum(jobs.est_worst_h[jobs.kind .== "obj"]), sum(jobs.est_worst_h[jobs.kind .== "grad"]))
@printf("jobs with worst-case < 5 min: %d\n", count(<(5 / 60), jobs.est_worst_h))
