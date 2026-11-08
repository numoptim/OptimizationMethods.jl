# Gate-2 spot check: re-run selected runs by calling the solvers directly (no timing
# layer, no budgets) with threshold = the 1e-3 tolerance, and compare with the
# recorded 1e-3 hit (iteration, #F, #grad, F at the hit iterate).
#   julia --project=. src/spot_check.jl [merged_dir]
include(joinpath(@__DIR__, "runner.jl"))
using Test

merged = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "results", "merged")
runs = CSV.read(joinpath(merged, "runs.csv"), DataFrame)
hits = CSV.read(joinpath(merged, "hits.csv"), DataFrame)
problems = CSV.read(joinpath(@__DIR__, "..", "config", "problems.csv"), DataFrame)
starts = CSV.read(joinpath(@__DIR__, "..", "config", "starts.csv"), DataFrame)
algs = CSV.read(joinpath(@__DIR__, "..", "config", "algorithms.csv"), DataFrame)

@testset "spot check against direct solver calls" begin
    for method in ("OURS", "ARM", "ADATR")
        # first run of this method with a 1e-3 hit, smallest problem first
        cand = innerjoin(hits[(hits.tol .== 1e-3) .& (hits.hit .== true), :],
            runs[runs.method .== method, [:problem_id, :start_id, :algorithm_id]],
            on = [:problem_id, :start_id, :algorithm_id])
        sort!(cand, [:problem_id, :start_id, :algorithm_id])
        @test nrow(cand) > 0
        h = cand[1, :]
        alg = algs[algs.algorithm_id .== h.algorithm_id, :][1, :]
        pr = problems[problems.problem_id .== h.problem_id, :][1, :]
        st = starts[(starts.problem_id .== h.problem_id) .& (starts.start_id .== h.start_id), :][1, :]
        prob = build_problem(pr.V, pr.n, pr.m, pr.seed)
        x0 = build_start(pr.n, st.seed)
        optData, solver = make_algorithm(alg, copy(x0), 1e-3, 100_000)
        NLPModels.reset!(prob.counters)
        x = solver(optData, prob)
        c = base_counters(prob.counters)
        n_grad_direct, n_obj_direct = Int(c.neval_grad), Int(c.neval_obj)   # before our own F evaluation
        precomp, store = OM.initialize(prob)
        Fx = OM.obj(prob, precomp, store, x)
        println("$method: problem $(h.problem_id) start $(h.start_id) alg $(h.algorithm_id): direct n_grad=$(n_grad_direct) n_obj=$(n_obj_direct) iter=$(optData.stop_iteration); recorded n_grad=$(h.n_grad) n_obj=$(h.n_obj) iter=$(h.iteration)")
        @test n_grad_direct == h.n_grad
        @test n_obj_direct == h.n_obj
        @test optData.stop_iteration == h.iteration
        @test Fx ≈ h.F_at_hit rtol = 1e-10
    end
end
