# Usage (from experiments/coap-revision):
#   julia --project=. src/run_job.jl JOB_ID [--jobs config/jobs.csv] [--out results/raw]
# Runs every (problem, start, algorithm) of the job and writes
#   <out>/job_<id>/{runs,hits,ours_events}.csv   (written to a temp dir, then renamed).
include(joinpath(@__DIR__, "runner.jl"))

function main(args)
    isempty(args) && error("usage: run_job.jl JOB_ID [--jobs FILE] [--out DIR]")
    job_id = parse(Int, args[1])
    root = joinpath(@__DIR__, "..")
    jobs_file = joinpath(root, "config", "jobs.csv")
    out_dir = joinpath(root, "results", "raw")
    i = 2
    while i <= length(args)
        if args[i] == "--jobs"; jobs_file = args[i+1]; i += 2
        elseif args[i] == "--out"; out_dir = args[i+1]; i += 2
        else error("unknown argument $(args[i])") end
    end
    jobs = CSV.read(jobs_file, DataFrame)
    problems = CSV.read(joinpath(root, "config", "problems.csv"), DataFrame)
    starts = CSV.read(joinpath(root, "config", "starts.csv"), DataFrame)
    algs = CSV.read(joinpath(root, "config", "algorithms.csv"), DataFrame)
    job = jobs[jobs.job_id .== job_id, :]
    nrow(job) == 1 || error("job $job_id not found (or duplicated) in $jobs_file")
    job = job[1, :]

    final_dir = joinpath(out_dir, "job_$(job_id)")
    isdir(final_dir) && error("$final_dir already exists; refusing to overwrite")
    tmp_dir = joinpath(out_dir, ".job_$(job_id).tmp")
    rm(tmp_dir; force = true, recursive = true)
    mkpath(tmp_dir)

    pr = problems[problems.problem_id .== job.problem_id, :][1, :]
    prob = build_problem(pr.V, pr.n, pr.m, pr.seed)
    # detect RNG drift between machines / Julia patch versions
    cs = data_checksums(prob)
    ref = (pr.sum_design, pr.sumsq_design, pr.sum_response, pr.sumsq_response)
    all(isapprox.(cs, ref; rtol = 1e-10)) ||
        error("data checksums for problem $(pr.problem_id) differ from config/problems.csv " *
              "(RNG drift?): got $cs, expected $ref")

    G_max, F_max = Int(job.G_max), Int(job.F_max)
    runs, hits, events = NamedTuple[], NamedTuple[], NamedTuple[]
    errors = String[]
    t_job = time()
    for s in parse_ids(job.start_ids), a in parse_ids(job.algorithm_ids)
        alg = algs[algs.algorithm_id .== a, :][1, :]
        st = starts[(starts.problem_id .== pr.problem_id) .& (starts.start_id .== s), :][1, :]
        x0 = build_start(pr.n, st.seed)
        timeout_s = Float64(alg.uses_obj ? job.timeout_obj_s : job.timeout_grad_s)
        res = run_one(prob, alg, x0; G_max = G_max, F_max = F_max, timeout_s = timeout_s,
            problem_row = pr, start_id = s, job_id = job_id)
        push!(runs, res.run); append!(hits, res.hits); append!(events, res.events)
        isempty(res.errmsg) || push!(errors, "problem $(pr.problem_id) start $s algorithm $a: $(res.errmsg)")
        @printf("[job %d] problem %d start %d alg %3d %-7s %-15s  grad=%d obj=%d  %.1fs (elapsed %.0fs)\n",
            job_id, pr.problem_id, s, a, alg.method, res.run.status, res.run.n_grad,
            res.run.n_obj, res.run.time_total_s, time() - t_job)
        flush(stdout)
    end
    CSV.write(joinpath(tmp_dir, "runs.csv"), DataFrame(runs))
    CSV.write(joinpath(tmp_dir, "hits.csv"), DataFrame(hits))
    ev_df = isempty(events) ?
        DataFrame(problem_id = Int[], start_id = Int[], algorithm_id = Int[], k = Int[],
            j_k = Int[], ball = Bool[], lower = Bool[], upper = Bool[], maxiter = Bool[],
            accepted = Union{Missing,Bool}[]) : DataFrame(events)
    CSV.write(joinpath(tmp_dir, "ours_events.csv"), ev_df)
    isempty(errors) || write(joinpath(tmp_dir, "errors.log"), join(errors, "\n") * "\n")
    mv(tmp_dir, final_dir)
    println("[job $job_id] done: $(length(runs)) runs, $(length(errors)) errors, $(round(time() - t_job; digits = 1)) s")
end

main(ARGS)
