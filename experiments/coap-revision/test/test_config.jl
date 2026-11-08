# Checks config/{problems,starts,algorithms}.csv. Run from experiments/coap-revision:
#   julia --project=. test/test_config.jl
using Test, CSV, DataFrames, Random
include(joinpath(@__DIR__, "..", "src", "instances.jl"))

cfg(f) = joinpath(@__DIR__, "..", "config", f)
problems = CSV.read(cfg("problems.csv"), DataFrame)
starts = CSV.read(cfg("starts.csv"), DataFrame)
algs = CSV.read(cfg("algorithms.csv"), DataFrame)

@testset "problems.csv" begin
    @test nrow(problems) == 120
    @test problems.problem_id == collect(1:120)
    @test allunique(problems.seed)
    @test sort(unique(problems.V)) == VARIANCE_LABELS
    @test sort(unique(problems.n)) == N_VALUES && sort(unique(problems.m)) == M_VALUES
    for V in VARIANCE_LABELS, n in N_VALUES, m in M_VALUES
        d = problems[(problems.V .== V) .& (problems.n .== n) .& (problems.m .== m), :]
        @test sort(d.replicate) == collect(1:N_REPLICATES)
    end
    # the paper's design checks
    @test all(problems.checks_y_in_01)
    @test all(problems.checks_mu_in_03_07)
    @test all(problems.checks_noise_lt_03)
    @test all(problems.checks_design_regenerated)
    # regenerating a sample of problems reproduces the recorded data exactly
    for p in (1, 17, 40, 41, 80, 120)
        r = problems[p, :]
        prob = build_problem(r.V, r.n, r.m, r.seed)
        cs = data_checksums(prob)
        @test cs[1] ≈ r.sum_design rtol = 1e-12
        @test cs[2] ≈ r.sumsq_design rtol = 1e-12
        @test cs[3] ≈ r.sum_response rtol = 1e-12
        @test cs[4] ≈ r.sumsq_response rtol = 1e-12
        @test size(prob.design) == (r.m, r.n)
    end
end

@testset "starts.csv" begin
    @test nrow(starts) == 1200
    @test allunique(starts.seed)
    for p in 1:120
        @test sort(starts.start_id[starts.problem_id .== p]) == collect(1:N_STARTS)
    end
    for i in (1, 500, 1200)
        r = starts[i, :]
        n = problems.n[r.problem_id]
        x0 = build_start(n, r.seed)
        @test length(x0) == n
        @test all(-10 .<= x0 .<= 10)
        @test minimum(x0) ≈ r.x0_min && maximum(x0) ≈ r.x0_max
        @test x0 == build_start(n, r.seed)       # reproducible
    end
end

@testset "algorithms.csv" begin
    @test nrow(algs) == 100
    @test algs.algorithm_id == collect(1:100)
    counts = Dict(m => count(==(m), algs.method) for m in unique(algs.method))
    @test counts == Dict("FIXED" => 7, "DIM" => 7, "BBGD" => 14, "LIPAPP" => 7,
        "NEST" => 7, "WEIGHT" => 7, "OURS" => 1, "ARM" => 1, "WATCH" => 7,
        "SPEC" => 14, "ADATR" => 14, "ASTR1" => 14)
    @test sum(counts[m] for m in ("FIXED","DIM","BBGD","LIPAPP","NEST","WEIGHT","OURS","ARM","WATCH")) == 58
    # F is only used by OURS, ARM, WATCH, SPEC
    @test Set(algs.method[algs.uses_obj]) == Set(["OURS", "ARM", "WATCH", "SPEC"])
    etas = sort(unique(skipmissing(algs.eta)))
    @test etas == [1e-4, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    @test nrow(algs) * nrow(starts) == 120_000
    # every struct/solver exists in the package
    for r in eachrow(algs)
        @test isdefined(OptimizationMethods, Symbol(r.struct_name))
        @test isdefined(OptimizationMethods, Symbol(r.solver))
    end
end
