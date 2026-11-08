# Merges results/raw/job_*/ into results/merged/{runs,hits,ours_events}.csv and validates.
#   julia --project=. src/collect.jl [--pilot] [--jobs FILE] [--raw DIR] [--out DIR]
# Default: all jobs in config/jobs.csv must be present and complete (120,000 runs).
# --pilot: only the job directories that are present must be complete (subset runs).
# Writes <out>/resubmit_jobs.txt (job ids that are missing or invalid) and exits
# nonzero on any failure.
using CSV, DataFrames, Printf, Statistics

const ROOT = joinpath(@__DIR__, "..")
const ALLOWED_STATUS = ["converged_1e-9", "budget_grad", "budget_obj", "timeout",
    "nan_or_inf", "error", "stalled"]
const TOLS = [1e-3, 1e-5, 1e-7, 1e-9]
const NO_OBJ_METHODS = ["FIXED", "DIM", "BBGD", "LIPAPP", "NEST", "WEIGHT", "ADATR", "ASTR1"]
const RUN_COLS = [:problem_id, :n, :m, :V, :replicate, :start_id, :algorithm_id, :method,
    :eta, :long_stepsize, :adatr_alpha, :astr1_variant, :status, :iterations, :n_obj, :n_grad,
    :time_total_s, :time_obj_s, :time_grad_s, :F0, :F_final, :gradnorm0, :gradnorm_final,
    :julia_version, :hostname, :cpu_model, :job_id]
const HIT_COLS = [:problem_id, :start_id, :algorithm_id, :tol, :hit, :iteration, :n_obj, :n_grad,
    :time_total_s, :time_obj_s, :time_grad_s, :gradnorm, :F_at_hit, :reliable]
const EVENT_COLS = [:problem_id, :start_id, :algorithm_id, :k, :j_k, :ball, :lower, :upper,
    :maxiter, :accepted]

parse_ids(s) = isempty(strip(string(s))) ? Int[] : parse.(Int, split(string(s), ';'))
bad(x) = ismissing(x) || (x isa AbstractFloat && !isfinite(x))

function main(args)
    pilot = "--pilot" in args
    opt(name, default) = (i = findfirst(==(name), args); i === nothing ? default : args[i+1])
    jobs_file = opt("--jobs", joinpath(ROOT, "config", "jobs.csv"))
    raw_dir = opt("--raw", joinpath(ROOT, "results", "raw"))
    out_dir = opt("--out", joinpath(ROOT, "results", "merged"))
    mkpath(out_dir)
    jobs = CSV.read(jobs_file, DataFrame)
    algs = CSV.read(joinpath(ROOT, "config", "algorithms.csv"), DataFrame)
    method_of = Dict(algs.algorithm_id .=> algs.method)

    failures = String[]
    fail!(msg) = (push!(failures, msg); println("FAIL: ", msg))
    resubmit = Set{Int}()

    present = Int[]
    for d in readdir(raw_dir)
        m = match(r"^job_(\d+)$", d)
        m === nothing || push!(present, parse(Int, m.captures[1]))
    end
    sort!(present)
    expected_jobs = pilot ? intersect(jobs.job_id, present) : collect(jobs.job_id)
    unknown = setdiff(present, jobs.job_id)
    isempty(unknown) || fail!("job directories not in $(basename(jobs_file)): $(unknown[1:min(end,10)])")
    missing_jobs = setdiff(expected_jobs, present)
    if !isempty(missing_jobs)
        fail!("$(length(missing_jobs)) job(s) missing")
        union!(resubmit, missing_jobs)
    end

    runs_all, hits_all, ev_all = DataFrame[], DataFrame[], DataFrame[]
    for j in intersect(expected_jobs, present)
        dir = joinpath(raw_dir, "job_$j")
        files = [joinpath(dir, f) for f in ("runs.csv", "hits.csv", "ours_events.csv")]
        if !all(isfile, files)
            fail!("job $j: missing output file(s)"); push!(resubmit, j); continue
        end
        r = CSV.read(files[1], DataFrame; types = Dict(:eta => Union{Missing,Float64},
            :long_stepsize => Union{Missing,Bool}, :adatr_alpha => Union{Missing,Float64},
            :astr1_variant => Union{Missing,String}))
        h = CSV.read(files[2], DataFrame)
        e = CSV.read(files[3], DataFrame)
        # per-job completeness
        job = jobs[jobs.job_id .== j, :][1, :]
        exp = Set((job.problem_id, s, a) for s in parse_ids(job.start_ids), a in parse_ids(job.algorithm_ids))
        got = [(row.problem_id, row.start_id, row.algorithm_id) for row in eachrow(r)]
        if length(got) != length(exp) || Set(got) != exp || length(unique(got)) != length(got)
            fail!("job $j: runs do not match the job specification ($(length(got)) rows, expected $(length(exp)))")
            push!(resubmit, j); continue
        end
        if nrow(h) != 4 * nrow(r)
            fail!("job $j: $(nrow(h)) hit rows, expected $(4 * nrow(r))"); push!(resubmit, j); continue
        end
        if !isempty(setdiff(Symbol.(names(r)), RUN_COLS)) || !isempty(setdiff(RUN_COLS, Symbol.(names(r))))
            fail!("job $j: runs.csv columns differ from the schema"); push!(resubmit, j); continue
        end
        push!(runs_all, r); push!(hits_all, h); push!(ev_all, e)
    end
    isempty(runs_all) && (println("no valid job output found"); exit(1))
    runs = vcat(runs_all...)
    hits = vcat(hits_all...)
    events = vcat(ev_all...; cols = :union)
    key(df) = collect(zip(df.problem_id, df.start_id, df.algorithm_id))

    # global uniqueness
    if length(unique(key(runs))) != nrow(runs)
        fail!("duplicate (problem, start, algorithm) rows across jobs")
    end
    # expected total for a full collection
    if !pilot
        total = sum(jobs.n_runs)
        nrow(runs) == total || fail!("runs.csv has $(nrow(runs)) rows, expected $total")
        if jobs_file == joinpath(ROOT, "config", "jobs.csv") && total != 120_000
            fail!("config/jobs.csv describes $total runs, expected 120000")
        end
    end

    # statuses
    badstat = runs[.!in.(runs.status, Ref(ALLOWED_STATUS)), :]
    nrow(badstat) == 0 || fail!("$(nrow(badstat)) rows with a status outside $(ALLOWED_STATUS)")

    # NaN/Inf/missing in counts and times
    for c in (:iterations, :n_obj, :n_grad, :time_total_s, :time_obj_s, :time_grad_s)
        nb = count(bad, runs[!, c])
        if nb > 0
            fail!("runs.csv: $nb NaN/Inf/missing values in $c")
            union!(resubmit, unique(runs.job_id[bad.(runs[!, c])]))
        end
    end
    hh = hits[hits.hit .== true, :]
    for c in (:iteration, :n_obj, :n_grad, :time_total_s, :time_obj_s, :time_grad_s)
        nb = count(bad, hh[!, c])
        nb == 0 || fail!("hits.csv: $nb NaN/Inf/missing values in $c (hit rows)")
    end

    # explicit hit = false rows: exactly 4 tolerances per run
    sort!(hits, [:problem_id, :start_id, :algorithm_id, :tol])
    grp = groupby(hits, [:problem_id, :start_id, :algorithm_id])
    nrow_wrong = 0; nonmono = 0; wrongtol = 0
    for g in grp
        if nrow(g) != 4 || sort(g.tol) != sort(TOLS)
            wrongtol += 1; continue
        end
        # g sorted by tol ascending: 1e-9 first. Reverse to loose -> tight.
        gg = g[reverse(1:4), :]
        h = gg.hit
        # hits must be non-increasing along loose -> tight: once false, stays false
        ok = true
        for i in 2:4
            h[i] && !h[i-1] && (ok = false)
        end
        # hit quantities non-decreasing for consecutive hit tolerances
        for i in 2:4
            if h[i] && h[i-1]
                for c in (:iteration, :n_obj, :n_grad, :time_total_s, :time_obj_s, :time_grad_s)
                    gg[i, c] < gg[i-1, c] && (ok = false)
                end
            end
        end
        ok || (nonmono += 1)
    end
    wrongtol == 0 || fail!("$wrongtol runs without exactly 4 tolerance rows")
    nonmono == 0 || fail!("$nonmono runs with hit data not monotone across tolerances")

    # #F = 0 for methods that never evaluate F
    nof = runs[in.(runs.method, Ref(NO_OBJ_METHODS)) .& (runs.n_obj .!= 0), :]
    nrow(nof) == 0 || fail!("$(nrow(nof)) runs of gradient-only methods with n_obj != 0")
    hnof = hits[in.([method_of[a] for a in hits.algorithm_id], Ref(NO_OBJ_METHODS)) .& (coalesce.(hits.n_obj, 0) .!= 0), :]
    nrow(hnof) == 0 || fail!("$(nrow(hnof)) hit rows of gradient-only methods with n_obj != 0")

    # budgets respected, and status consistent with the 1e-9 hit
    jb = Dict(jobs.job_id .=> zip(jobs.G_max, jobs.F_max))
    over = count(i -> runs.n_grad[i] > jb[runs.job_id[i]][1] || runs.n_obj[i] > jb[runs.job_id[i]][2], 1:nrow(runs))
    over == 0 || fail!("$over runs exceed their evaluation budget")
    h9 = hits[hits.tol .== 1e-9, :]
    hit9 = Dict(zip(key(h9), h9.hit))
    inconsistent = count(i -> (runs.status[i] == "converged_1e-9") != hit9[(runs.problem_id[i], runs.start_id[i], runs.algorithm_id[i])], 1:nrow(runs))
    inconsistent == 0 || fail!("$inconsistent runs where status 'converged_1e-9' disagrees with the 1e-9 hit")

    # reliable must imply hit
    nrow(hits[(hits.reliable .== true) .& (hits.hit .== false), :]) == 0 || fail!("reliable without hit")

    # write merged files
    CSV.write(joinpath(out_dir, "runs.csv"), runs[:, RUN_COLS])
    CSV.write(joinpath(out_dir, "hits.csv"), hits[:, HIT_COLS])
    ev_cols = [c for c in EVENT_COLS if c in Symbol.(names(events))]
    CSV.write(joinpath(out_dir, "ours_events.csv"), events[:, ev_cols])
    write(joinpath(out_dir, "resubmit_jobs.txt"), join(sort(collect(resubmit)), "\n") * (isempty(resubmit) ? "" : "\n"))

    # summary
    println("\n==== summary ====")
    @printf("jobs read: %d of %d expected   runs: %d   hit rows: %d   OURS inner loops: %d\n",
        length(runs_all), length(expected_jobs), nrow(runs), nrow(hits), nrow(events))
    println(sort(combine(groupby(runs, :status), nrow => :runs), :runs, rev = true))
    errs = runs[runs.status .== "error", :]
    nrow(errs) == 0 || println("runs with status 'error': $(nrow(errs)) (see errors.log in the job directories)")
    println("reliability at each tolerance (all methods pooled): ",
        join([@sprintf("%.0e: %.3f", t, mean(hits.reliable[hits.tol .== t])) for t in TOLS], "  "))
    println("jobs to resubmit: ", isempty(resubmit) ? "none" : join(sort(collect(resubmit)), " "))
    if isempty(failures)
        println("\nVALID")
    else
        println("\n$(length(failures)) validation failure(s); see $(joinpath(out_dir, "resubmit_jobs.txt"))")
        exit(1)
    end
end

main(ARGS)
