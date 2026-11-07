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
end

end
