# Quick structural check of results/merged/*.csv before analysing (no raw data needed).
#   julia --project=analysis src/analysis/check_merged.jl [merged_dir] [--pilot]
include(joinpath(@__DIR__, "common.jl"))
using Test, Printf
dir = isempty(filter(!startswith("--"), ARGS)) ? joinpath(@__DIR__, "..", "..", "results", "merged") : filter(!startswith("--"), ARGS)[1]
pilot = "--pilot" in ARGS
runs, hits, events = load_merged(dir)
allowed = ["converged_1e-9", "budget_grad", "budget_obj", "timeout", "nan_or_inf", "error", "stalled"]
@testset "merged CSVs" begin
    pilot || @test nrow(runs) == 120_000
    @test nrow(hits) == 4 * nrow(runs)
    @test all(in(allowed), runs.status)
    @test allunique(zip(runs.problem_id, runs.start_id, runs.algorithm_id))
    @test sort(unique(hits.tol)) == sort(TOLS)
    @test all(in([true, false]), hits.hit)
    @test all(x -> !ismissing(x) && isfinite(x), runs.n_grad) && all(x -> !ismissing(x) && isfinite(x), runs.n_obj)
    @test all(runs.n_obj[in.(runs.method, Ref(["FIXED","DIM","BBGD","LIPAPP","NEST","WEIGHT","ADATR","ASTR1"]))] .== 0)
    @test all(in(runs.method), ALL_METHODS) || pilot
end
n_loops = countlines(joinpath(dir, "ours_events.csv")) - 1      # streamed: the file can have ~100M rows
println("rows: runs=$(nrow(runs)) hits=$(nrow(hits)) ours_events=$n_loops ours_events_summary=$(nrow(events))")
println(combine(groupby(runs, :status), nrow => :runs))
