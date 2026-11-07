module TestExperimentRecorderSolvers

using Test, OptimizationMethods, Random, LinearAlgebra, NLPModels

# For every solver used in the COAP experiments: the hit recorded at tolerance ε
# during one run (threshold smaller than ε) equals a separate run with
# threshold = ε: same #F, #∇F, iteration and iterate.
@testset "Experiment recorder: all solvers" begin
    Random.seed!(77)
    T = Float64
    n = 5
    prob = OptimizationMethods.LeastSquares(T, nequ = 40, nvar = n)
    L = opnorm(prob.coef' * prob.coef)
    x0 = randn(T, n)
    tols = [1e-1, 1e-3]
    thr_small = 1e-5
    maxit = 20_000
    stepdown = OptimizationMethods.stepdown_100_step_size

    # name => (constructor of solver given threshold, solver function, has_F)
    cases = Dict(
        "BBGD long" => (thr -> BarzilaiBorweinGD(T; x0 = copy(x0), init_stepsize = 1/L,
            long_stepsize = true, threshold = thr, max_iterations = maxit), barzilai_borwein_gd),
        "BBGD short" => (thr -> BarzilaiBorweinGD(T; x0 = copy(x0), init_stepsize = 1/L,
            long_stepsize = false, threshold = thr, max_iterations = maxit), barzilai_borwein_gd),
        "DIM" => (thr -> DiminishingStepGD(T; x0 = copy(x0), step_size_function = stepdown,
            step_size_scaling = 1/L, threshold = thr, max_iterations = maxit), diminishing_step_gd),
        "LIPAPP" => (thr -> LipschitzApproxGD(T; x0 = copy(x0), init_stepsize = 1/L,
            threshold = thr, max_iterations = maxit), lipschitz_approximation_gd),
        "NEST" => (thr -> NesterovAcceleratedGD(T; x0 = copy(x0), step_size = 1/L,
            threshold = thr, max_iterations = maxit), nesterov_accelerated_gd),
        "WEIGHT" => (thr -> WeightedNormDampingGD(T; x0 = copy(x0),
            init_norm_damping_factor = L, threshold = thr, max_iterations = maxit),
            weighted_norm_damping_gd),
        "ARM" => (thr -> BacktrackingGD(T; x0 = copy(x0), α = T(1), δ = T(0.5), ρ = T(1e-4),
            line_search_max_iteration = 100, threshold = thr, max_iterations = maxit),
            backtracking_gd),
        "WATCH" => (thr -> WatchdogFixedSimpleGD(T; x0 = copy(x0), α = T(1), δ = T(0.5),
            ρ = T(1e-4), line_search_max_iterations = 100, window_size = 1,
            inner_loop_max_iterations = 100, threshold = thr, max_iterations = maxit),
            watchdog_fixed_simple_gd),
        "SPEC" => (thr -> SafeBarzilaiBorweinNLSMaxValGD(T; x0 = copy(x0), δ = T(0.5),
            ρ = T(1e-4), window_size = 10, line_search_max_iteration = 100,
            init_stepsize = T(1), long_stepsize = true, α_lower = T(1e-10),
            α_default = T(1), threshold = thr, max_iterations = maxit),
            safe_barzilai_borwein_nls_maxval_gd),
        "ADATR a0.9" => (thr -> AdaptiveTrustRegionGD(T; x0 = copy(x0), init_scaling = L,
            α = T(0.9), b_min = T(1e-4), threshold = thr, max_iterations = maxit),
            adaptive_trust_region_gd),
        "ADATR a0" => (thr -> AdaptiveTrustRegionGD(T; x0 = copy(x0), init_scaling = L,
            α = T(0.0), b_min = T(1e-4), threshold = thr, max_iterations = maxit),
            adaptive_trust_region_gd),
        "ASTR1 adagrad" => (thr -> ASTR1GD(T; x0 = copy(x0), variant = :adagrad,
            θ = T(1), threshold = thr, max_iterations = maxit), astr1_gd),
        "ASTR1 maxg" => (thr -> ASTR1GD(T; x0 = copy(x0), variant = :maxg,
            θ = T(1), threshold = thr, max_iterations = maxit), astr1_gd),
        "OURS" => (thr -> NonsequentialArmijoAdaptiveGD(T; x0 = copy(x0), δ0 = T(1),
            δ_upper = T(1), ρ = T(1e-4), threshold = thr, max_iterations = maxit),
            nonsequential_armijo_adaptive_gd),
    )

    for name in sort(collect(keys(cases)))
        mk, run = cases[name]
        @testset "$name" begin
            timed = TimedProblem(prob; tols = tols)
            start!(timed)
            s = mk(thr_small)
            xend = run(s, timed)
            # wrapper does not change the result
            NLPModels.reset!(prob.counters)
            xplain = run(mk(thr_small), prob)
            @test xplain == xend
            for (i, ε) in enumerate(tols)
                h = timed.hits[i]
                @test h !== nothing
                h === nothing && continue
                NLPModels.reset!(prob.counters)
                s_sep = mk(ε)
                x_sep = run(s_sep, prob)
                c = prob.counters.counters
                @test h.n_obj == c.neval_obj
                @test h.n_grad == c.neval_grad
                @test h.iteration == s_sep.stop_iteration
                @test h.x == x_sep
                @test h.gradnorm <= ε
            end
            for i in 1:length(tols)-1
                @test timed.hits[i].n_grad <= timed.hits[i+1].n_grad
                @test timed.hits[i].n_obj <= timed.hits[i+1].n_obj
                @test timed.hits[i].time_total_ns <= timed.hits[i+1].time_total_ns
            end
            @test timed.time_grad_ns > 0
        end
    end

    # Budget stopping: exact counts, clean stop (also mid line search / inner loop)
    @testset "Budgets" begin
        has_obj = ["ARM", "WATCH", "SPEC", "OURS"]
        for name in sort(collect(keys(cases)))
            mk, run = cases[name]
            # unbudgeted reference
            ref = TimedProblem(prob); start!(ref)
            sref = mk(1e-8)
            xref, st = run_solver!(run, sref, ref)
            @test st == :finished
            Gref = ref.counters.counters.neval_grad
            Fref = ref.counters.counters.neval_obj
            @test Gref > 6

            # no binding budget => same result
            big = TimedProblem(prob; max_grad = 10 * Gref + 10, max_obj = 10 * Fref + 10)
            start!(big)
            xb, stb = run_solver!(run, mk(1e-8), big)
            @test stb == :finished && xb == xref

            # gradient budgets, including very small ones
            for G in (1, 2, 3, Gref ÷ 2)
                tp = TimedProblem(prob; max_grad = G); start!(tp)
                s = mk(1e-8)
                x, st = run_solver!(run, s, tp)
                @test st == :budget_grad
                @test tp.counters.counters.neval_grad == G
                @test all(isfinite, x)
                @test tp.last_iter >= 0
                # returned x is the last completed iterate stored by the solver
                @test x == s.iter_hist[tp.last_iter + 1]
            end

            # objective budgets
            if name in has_obj
                for F in (1, 2, max(Fref ÷ 2, 3))
                    tp = TimedProblem(prob; max_obj = F); start!(tp)
                    s = mk(1e-8)
                    x, st = run_solver!(run, s, tp)
                    @test st == :budget_obj
                    @test tp.counters.counters.neval_obj == F
                    @test all(isfinite, x)
                    @test tp.last_iter >= 0
                    @test x == s.iter_hist[tp.last_iter + 1]
                end
            else
                @test Fref == 0
            end

            # wall-clock cap
            tp = TimedProblem(prob; max_time_ns = 1); start!(tp)
            x, st = run_solver!(run, mk(1e-8), tp)
            @test st == :timeout
            @test tp.last_iter == -1 && all(isnan, x)
        end
    end
end

end
