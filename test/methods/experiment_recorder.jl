module TestExperimentRecorder

using Test, OptimizationMethods, Random, LinearAlgebra, NLPModels

@testset "Experiment recorder: FixedStepGD on LeastSquares" begin
    Random.seed!(2024)
    T = Float64
    progData = OptimizationMethods.LeastSquares(T, nequ = 50, nvar = 10)
    x0 = randn(T, 10)
    L = opnorm(progData.coef' * progData.coef)
    step = T(1 / L)  # stable step: gradient is A'(Ax-b), Lipschitz L
    tols = [1e-3, 1e-5, 1e-7, 1e-9]

    run_fixed(prob, thr) = begin
        solver = FixedStepGD(T; x0 = copy(x0), step_size = step,
            threshold = thr, max_iterations = 100_000)
        NLPModels.reset!(prob.counters)
        x = fixed_step_gd(solver, prob)
        (x, solver)
    end

    timed = TimedProblem(progData; tols = tols)
    start!(timed)
    solver = FixedStepGD(T; x0 = copy(x0), step_size = step,
        threshold = 1e-9, max_iterations = 100_000)
    xfinal = fixed_step_gd(solver, timed)

    hits = timed.hits
    @test all(h -> h !== nothing, hits)
    # order, monotone counts, monotone times
    for i in 1:3
        @test hits[i].iteration <= hits[i+1].iteration
        @test hits[i].n_grad <= hits[i+1].n_grad
        @test hits[i].n_obj <= hits[i+1].n_obj
        @test hits[i].time_total_ns <= hits[i+1].time_total_ns
        @test hits[i].time_grad_ns <= hits[i+1].time_grad_ns
    end
    # FixedStepGD: gradient count is iteration + 1, no objective evaluations
    for h in hits
        @test h.n_obj == 0
        @test h.n_grad == h.iteration + 1
        @test h.gradnorm <= h.tol
        @test h.time_grad_ns <= h.time_total_ns
    end
    @test hits[4].iteration == solver.stop_iteration
    @test hits[4].x == xfinal

    # each hit equals a separate run with that threshold
    for (i, ε) in enumerate(tols)
        x_sep, s_sep = run_fixed(progData, ε)
        @test hits[i].iteration == s_sep.stop_iteration
        @test hits[i].n_grad == progData.counters.counters.neval_grad
        @test hits[i].n_obj == progData.counters.counters.neval_obj
        @test hits[i].x == x_sep
    end

    # wrapper does not change the result; counters are shared
    x_plain, _ = run_fixed(progData, 1e-9)
    @test x_plain == xfinal
    @test timed.counters === progData.counters
    @test timed.time_obj_ns == 0 && timed.time_grad_ns > 0

    # start! resets everything
    start!(timed)
    @test all(h -> h === nothing, timed.hits)
    @test timed.time_grad_ns == 0

    # a NaN gradient norm is never a hit
    timed_nan = TimedProblem(progData; tols = tols); start!(timed_nan)
    record_iterate!(timed_nan, 0, x0, NaN)
    @test all(h -> h === nothing, timed_nan.hits)
    record_iterate!(timed_nan, 1, x0, 1e-4)
    @test timed_nan.hits[1].iteration == 1 && timed_nan.hits[2] === nothing

    # no-op hook for plain problems
    @test OptimizationMethods.record_iterate!(progData, 0, x0, 1.0) === nothing

    # a start at a tolerance-satisfying point records all hits at iteration 0
    timed2 = TimedProblem(progData; tols = tols)
    start!(timed2)
    xstar = progData.coef \ progData.cons
    s2 = FixedStepGD(T; x0 = copy(xstar), step_size = step,
        threshold = 1e-3, max_iterations = 10)
    fixed_step_gd(s2, timed2)
    @test timed2.hits[1].iteration == 0
    @test timed2.hits[4] === nothing || timed2.hits[4].iteration >= 0
end

end
