# Unit tests for the analysis functions on tiny synthetic data with hand-computed answers.
#   julia --project=analysis test/test_analysis.jl
using Test, DataFrames, Statistics
include(joinpath(@__DIR__, "..", "src", "analysis", "profiles.jl"))
include(joinpath(@__DIR__, "..", "src", "analysis", "tables.jl"))

@testset "build_table: weighted cost and joins" begin
    runs = DataFrame(problem_id = [1, 1], start_id = [1, 1], algorithm_id = [50, 1],
        method = ["OURS", "FIXED"], n_obj = [10, 0], n_grad = [100, 40],
        time_obj_s = [1.0, 0.0], time_grad_s = [0.1, 0.05])
    hits = DataFrame(problem_id = [1, 1, 1, 1], start_id = 1, algorithm_id = [50, 50, 1, 1],
        tol = [1e-3, 1e-5, 1e-3, 1e-5], hit = [true, false, true, true],
        iteration = [3, missing, 7, 9], n_obj = [2, missing, 0, 0], n_grad = [20, missing, 30, 40],
        time_total_s = [0.5, missing, 0.01, 0.02], reliable = [true, false, true, false])
    T = build_table(runs, hits)
    o = T[(T.algorithm_id .== 50) .& (T.tol .== 1e-3), :][1, :]
    @test o.c ≈ (1.0 / 10) / (0.1 / 100)            # = 100
    @test o.cost ≈ 2 * 100 + 20
    @test isnan(T.cost[(T.algorithm_id .== 50) .& (T.tol .== 1e-5)][1])
    f = T[(T.algorithm_id .== 1) .& (T.tol .== 1e-5), :][1, :]
    @test f.c == 0.0 && f.cost == 40 && f.iteration == 9
end

function synthetic()
    rows = NamedTuple[]
    add(p, a, m, rel, ng, no = 0.0; tol = 1e-3, tm = 1.0) =
        push!(rows, (problem_id = p, start_id = 1, algorithm_id = a, method = m, tol = tol,
            hit = rel, reliable = rel, iteration = 1.0, nobj = rel ? no : NaN, ngrad = rel ? ng : NaN,
            cost = rel ? no * 10 + ng : NaN, time = rel ? tm : NaN, c = 10.0))
    # pair 1: OURS 10, ARM 20, FIXED(alg 3) fails, FIXED(alg 4) 7
    add(1, 50, "OURS", true, 10, 3); add(1, 51, "ARM", true, 20, 5)
    add(1, 3, "FIXED", false, 0); add(1, 4, "FIXED", true, 7)
    # pair 2: OURS 30, ARM 15, FIXED alg 3 15, alg 4 fails
    add(2, 50, "OURS", true, 30, 3); add(2, 51, "ARM", true, 15, 5)
    add(2, 3, "FIXED", true, 15); add(2, 4, "FIXED", false, 0)
    # pair 3: nobody succeeds
    add(3, 50, "OURS", false, 0); add(3, 51, "ARM", false, 0)
    add(3, 3, "FIXED", false, 0); add(3, 4, "FIXED", false, 0)
    return DataFrame(rows)
end

@testset "best configuration: reliability, then median #∇F" begin
    T = synthetic()
    bc = best_configs(T, 1e-3; methods = ["OURS", "ARM", "FIXED"])
    @test bc.method == ["OURS", "ARM", "FIXED"]
    # FIXED: alg 3 reliable on 1 pair (median #∇F 15), alg 4 on 1 pair (7): tie in reliability -> alg 4
    @test bc.algorithm_id[3] == 4
    @test bc.reliability ≈ [2 / 3, 2 / 3, 1 / 3]
end

@testset "performance profile" begin
    T = synthetic()
    cfg = Dict("OURS" => 50, "ARM" => 51, "FIXED" => 4)
    names, R, n, dropped = profile_ratios(T, cfg, 1e-3, :ngrad)
    @test names == ["FIXED", "OURS", "ARM"] || names == ["OURS", "ARM", "FIXED"]
    @test (n, dropped) == (2, 1)
    j = Dict(m => i for (i, m) in enumerate(names))
    # pair 1 costs: OURS 10, ARM 20, FIXED 7 -> ratios 10/7, 20/7, 1
    # pair 2 costs: OURS 30, ARM 15, FIXED Inf -> best 15 -> 2, 1, Inf
    @test R[1, j["OURS"]] ≈ 10 / 7 && R[1, j["ARM"]] ≈ 20 / 7 && R[1, j["FIXED"]] ≈ 1
    @test R[2, j["OURS"]] ≈ 2 && R[2, j["ARM"]] ≈ 1 && R[2, j["FIXED"]] == Inf
    ρ = profile_curves(R, [1.0, 2.0, 3.0])
    @test ρ[:, j["FIXED"]] ≈ [0.5, 0.5, 0.5]
    @test ρ[:, j["ARM"]] ≈ [0.5, 0.5, 1.0]    # ratios 20/7 and 1
    @test ρ[:, j["OURS"]] ≈ [0.0, 1.0, 1.0]
    # #F metric uses #F + 1
    _, Rf, _, _ = profile_ratios(T, cfg, 1e-3, :nobj)
    @test Rf[1, j["FIXED"]] ≈ 1.0 && Rf[1, j["OURS"]] ≈ 4.0 && Rf[1, j["ARM"]] ≈ 6.0
    taus = tau_grid(R)
    @test taus[1] == 1.0 && issorted(taus) && taus[end] >= 2.0
end

@testset "reliability table and margins" begin
    T = synthetic()
    bc = best_configs(T, 1e-3; methods = ["OURS", "ARM", "FIXED"])
    tab = reliability_table(T, bc, 1e-3, :ngrad)
    @test names(tab) == ["algorithm.name", "param", "total.x", "avg.reliability", "sd.reliability",
        "moe_plus.x", "moe_minus.x", "total.y", "avg.ngrad", "sd.ngrad", "moe_plus.y", "moe_minus.y"]
    o = tab[tab[!, "algorithm.name"] .== "OURS", :][1, :]
    rel = [1.0, 1.0, 0.0]
    @test o["total.x"] == 3 && o["avg.reliability"] ≈ 2 / 3
    @test o["moe_plus.x"] ≈ 2 / 3 + 1.96 * std(rel) / sqrt(3)
    @test o["total.y"] == 2 && o["avg.ngrad"] ≈ 20
    @test o["moe_minus.y"] ≈ 20 - 1.96 * std([10.0, 30.0]) / sqrt(2)
    @test moe95([1.0]) == 0.0
end

@testset "OURS events up to the hit" begin
    T = DataFrame(problem_id = [1, 2], start_id = 1, algorithm_id = 50, method = "OURS",
        tol = 1e-3, hit = true, reliable = true, iteration = [3.0, 2.0],
        nobj = 0.0, ngrad = 0.0, cost = 0.0, time = 0.0, c = 0.0)
    ev = DataFrame(
        problem_id = [1, 1, 1, 1, 2, 2], start_id = 1, algorithm_id = 50,
        k = [1, 2, 3, 4, 1, 2], j_k = 5,
        ball = [false, false, false, true, false, false],
        lower = [true, true, false, false, true, true],
        upper = [false, false, true, false, false, false],
        maxiter = false,
        accepted = [true, true, false, true, true, true])
    # per-run counts over the loops with k <= hit iteration (what collect.jl's event_summary writes)
    summ = DataFrame(problem_id = [1, 2], start_id = 1, algorithm_id = 50, tol = 1e-3,
        loops = [3, 2], ball = 0, lower = [2, 2], upper = [1, 0], maxiter = 0,
        acc_known = [3, 2], acc_rejected = [1, 0])
    tab = events_table(summ, 1e-3)
    p = tab[!, "Average.Proportion.on.Inner.Loops"]
    @test tab.Events[1:4] == ["Bounding Ball Event", "Lower Gradient Interval", "Upper Gradient Interval", "Max Iteration"]
    # run 1 (k <= 3): lower 2/3, upper 1/3; run 2: lower 100
    @test p[1] ≈ 0.0
    @test p[2] ≈ (200 / 3 + 100) / 2
    @test p[3] ≈ (100 / 3 + 0) / 2
    @test p[5] ≈ (100 / 3 + 0) / 2             # rejection: run 1 has 1 of 3 rejected, run 2 none
    @test tab.runs[1] == 2 && tab.loops[1] == 5
end
