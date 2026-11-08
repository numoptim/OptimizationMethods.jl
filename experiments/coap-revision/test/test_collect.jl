# Negative tests for src/collect.jl: tamper with copies of the local pilot output
# (results/pilot_raw, produced by the Gate-2 pilot) and check collect.jl exits nonzero.
#   julia --project=. test/test_collect.jl
using Test, CSV, DataFrames
const ROOT = joinpath(@__DIR__, "..")
const RAW = joinpath(ROOT, "results", "pilot_raw")
const JOBS = joinpath(ROOT, "config", "pilot_jobs.csv")
if !isdir(RAW)
    @info "results/pilot_raw not found; skipping collect tests"
    exit(0)
end
const JULIA = joinpath(Sys.BINDIR, Base.julia_exename())

function run_collect(raw, out)
    cmd = `$JULIA --project=$ROOT $(joinpath(ROOT, "src", "collect.jl")) --pilot --jobs $JOBS --raw $raw --out $out`
    p = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = devnull))
    return p.exitcode
end

function tampered(f)
    raw = mktempdir(); out = mktempdir()
    cp(RAW, joinpath(raw, "raw"); force = true)
    f(joinpath(raw, "raw"))
    return run_collect(joinpath(raw, "raw"), out)
end

function edit_csv(path, f)
    df = CSV.read(path, DataFrame)
    f(df)
    CSV.write(path, df)
end

@testset "collect.jl validation" begin
    @test run_collect(RAW, mktempdir()) == 0                      # clean data is valid
    # missing job output file
    @test tampered(raw -> rm(joinpath(raw, "job_2", "hits.csv"))) != 0
    # duplicated run row
    @test tampered(raw -> edit_csv(joinpath(raw, "job_1", "runs.csv"),
        df -> push!(df, df[1, :]))) != 0
    # invalid status
    @test tampered(raw -> edit_csv(joinpath(raw, "job_1", "runs.csv"),
        df -> df.status[1] = "weird")) != 0
    # NaN in a count/time column
    @test tampered(raw -> edit_csv(joinpath(raw, "job_1", "runs.csv"),
        df -> (df.time_total_s = Vector{Union{Missing,Float64}}(df.time_total_s); df.time_total_s[2] = NaN))) != 0
    # gradient-only method with F evaluations
    @test tampered(raw -> edit_csv(joinpath(raw, "job_1", "runs.csv"),
        df -> df.n_obj[1] = 5)) != 0       # job 1 starts with FIXED
    # hit counts decreasing across tolerances: make the 1e-9 hit earlier than the 1e-3 hit
    @test tampered(raw -> edit_csv(joinpath(raw, "job_1", "hits.csv"), df -> begin
        i = findfirst((df.tol .== 1e-9) .& (df.hit .== true))
        df.n_grad[i] = 0
    end)) != 0
    # status says converged but no 1e-9 hit
    @test tampered(raw -> edit_csv(joinpath(raw, "job_1", "runs.csv"), df -> begin
        i = findfirst(df.status .== "budget_grad"); df.status[i] = "converged_1e-9"
    end)) != 0
end
