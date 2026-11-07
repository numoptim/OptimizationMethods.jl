module TestGDASTR1

using Test, OptimizationMethods, Random, LinearAlgebra, NLPModels

# f(x) = 1/2 ||A x||^2 with A = diag(1, sqrt(10)): f = 1/2 x' diag(1,10) x
quad_problem() = OptimizationMethods.LeastSquares(
    Matrix(Diagonal([1.0, sqrt(10.0)])), zeros(2); x0 = [1.0, 1.0])

@testset "Method: ASTR1 (first order)" begin
    Random.seed!(777)
    T = Float64

    ## struct and constructor
    @test supertype(ASTR1GD) == OptimizationMethods.AbstractOptimizerData
    for f in (:name, :variant, :θ, :ς, :threshold, :max_iterations, :iter_hist,
        :grad_val_hist, :weights, :accumulator, :stop_iteration)
        @test f in fieldnames(ASTR1GD)
    end
    s = ASTR1GD(T; x0 = zeros(3), variant = :adagrad, θ = 2.0, threshold = 1e-3,
        max_iterations = 7)
    @test s.ς == 0.01 && length(s.iter_hist) == 8 && s.stop_iteration == -1
    @test_throws AssertionError ASTR1GD(T; x0 = zeros(3), variant = :foo, θ = 1.0,
        threshold = 1.0, max_iterations = 1)
    @test_throws AssertionError ASTR1GD(T; x0 = zeros(3), variant = :maxg, θ = 0.0,
        threshold = 1.0, max_iterations = 1)

    ## pure weight functions
    w = zeros(2); acc = zeros(2)
    OptimizationMethods.astr1_adagrad_weights!(w, acc, [3.0, 4.0], 2.0, 0.01)
    @test acc == [9.0, 16.0]
    @test w ≈ 2.0 .* sqrt.([9.01, 16.01]) rtol = 1e-14
    OptimizationMethods.astr1_adagrad_weights!(w, acc, [1.0, 0.0], 2.0, 0.01)
    @test acc == [10.0, 16.0]
    w = zeros(2); acc = zeros(2)
    OptimizationMethods.astr1_maxg_weights!(w, acc, [3.0, -0.001], 4, 2.0, 0.01)
    @test acc == [3.0, 0.001]
    @test w ≈ 2.0 * 5.0^0.1 .* [3.0, 0.01] rtol = 1e-14
    OptimizationMethods.astr1_maxg_weights!(w, acc, [-5.0, 0.5], 5, 2.0, 0.01)
    @test acc == [5.0, 0.5]

    ## hand-computed iterations on f = 1/2 x' diag(1,10) x, x0 = (1,1), θ = 2
    θ, ς = 2.0, 0.01
    # gradient (x1, 10 x2)

    # --- adagrad: w_i = θ sqrt(ς + Σ g_i^2)
    g0 = (1.0, 10.0)
    w0 = (θ * sqrt(ς + g0[1]^2), θ * sqrt(ς + g0[2]^2))
    x1 = (1.0 - g0[1] / w0[1], 1.0 - g0[2] / w0[2])
    g1 = (x1[1], 10 * x1[2])
    w1 = (θ * sqrt(ς + g0[1]^2 + g1[1]^2), θ * sqrt(ς + g0[2]^2 + g1[2]^2))
    x2 = (x1[1] - g1[1] / w1[1], x1[2] - g1[2] / w1[2])
    g2 = (x2[1], 10 * x2[2])
    w2 = (θ * sqrt(ς + g0[1]^2 + g1[1]^2 + g2[1]^2),
          θ * sqrt(ς + g0[2]^2 + g1[2]^2 + g2[2]^2))
    x3 = (x2[1] - g2[1] / w2[1], x2[2] - g2[2] / w2[2])

    prob = quad_problem()
    s = ASTR1GD(T; x0 = [1.0, 1.0], variant = :adagrad, θ = θ, threshold = 0.0,
        max_iterations = 3)
    astr1_gd(s, prob)
    @test s.iter_hist[2] ≈ collect(x1) rtol = 1e-12
    @test s.iter_hist[3] ≈ collect(x2) rtol = 1e-12
    @test s.iter_hist[4] ≈ collect(x3) rtol = 1e-12
    @test s.weights ≈ collect(w2) rtol = 1e-12

    # --- maxg: w_i = θ (k+1)^(1/10) max(ς, max_j |g_ij|)
    mw0 = (θ * 1.0^0.1 * max(ς, abs(g0[1])), θ * 1.0^0.1 * max(ς, abs(g0[2])))
    y1 = (1.0 - g0[1] / mw0[1], 1.0 - g0[2] / mw0[2])
    h1 = (y1[1], 10 * y1[2])
    m1 = (max(abs(g0[1]), abs(h1[1])), max(abs(g0[2]), abs(h1[2])))
    mw1 = (θ * 2.0^0.1 * max(ς, m1[1]), θ * 2.0^0.1 * max(ς, m1[2]))
    y2 = (y1[1] - h1[1] / mw1[1], y1[2] - h1[2] / mw1[2])
    h2 = (y2[1], 10 * y2[2])
    m2 = (max(m1[1], abs(h2[1])), max(m1[2], abs(h2[2])))
    mw2 = (θ * 3.0^0.1 * max(ς, m2[1]), θ * 3.0^0.1 * max(ς, m2[2]))
    y3 = (y2[1] - h2[1] / mw2[1], y2[2] - h2[2] / mw2[2])

    prob = quad_problem()
    s = ASTR1GD(T; x0 = [1.0, 1.0], variant = :maxg, θ = θ, threshold = 0.0,
        max_iterations = 3)
    astr1_gd(s, prob)
    @test s.iter_hist[2] ≈ collect(y1) rtol = 1e-12
    @test s.iter_hist[3] ≈ collect(y2) rtol = 1e-12
    @test s.iter_hist[4] ≈ collect(y3) rtol = 1e-12
    @test s.weights ≈ collect(mw2) rtol = 1e-12

    ## invariants, objective evaluations, gradient counts
    A = randn(T, 50, 10) / sqrt(50)   # well-conditioned: eig(A'A) in about [0.3, 2.1]
    y = A * randn(T, 10) + 0.1 * randn(T, 50)
    ls = OptimizationMethods.LeastSquares(A, y; x0 = zeros(T, 10))
    grad_at(x) = A' * (A * x - y)
    for variant in (:adagrad, :maxg), θ in (1e4, 1.0, 1 / 10.0)
        NLPModels.reset!(ls.counters)
        s = ASTR1GD(T; x0 = randn(T, 10), variant = variant, θ = θ,
            threshold = 1e-6, max_iterations = 1000)
        astr1_gd(s, ls)
        K = s.stop_iteration
        c = ls.counters.counters
        @test c.neval_obj == 0
        @test c.neval_grad == K + 1

        # recompute the weights independently from the stored iterates
        sumsq = zeros(10); runmax = zeros(10)
        wprev = zeros(10)
        for k in 0:K-1
            gk = grad_at(s.iter_hist[k+1])
            sumsq .+= gk .^ 2
            runmax .= max.(runmax, abs.(gk))
            wk = variant == :adagrad ? θ .* sqrt.(0.01 .+ sumsq) :
                 θ * (k + 1)^0.1 .* max.(0.01, runmax)
            Δ = abs.(gk) ./ wk
            sk = s.iter_hist[k+2] - s.iter_hist[k+1]
            @test all(abs.(sk) .<= Δ .* (1 + 1e-10) .+ 1e-14)
            if variant == :adagrad
                @test all(wk .>= wprev)
            end
            @test all(wk .>= θ * 0.01)   # AS.4-type lower bound
            wprev = wk
        end
        K > 0 && @test s.weights ≈ wprev rtol = 1e-10
    end

    ## convergence on a well-conditioned 50x10 least squares, η = 1 (θ = 1/η)
    for variant in (:adagrad, :maxg)
        NLPModels.reset!(ls.counters)
        s = ASTR1GD(T; x0 = randn(T, 10), variant = variant, θ = 1.0,
            threshold = 1e-6, max_iterations = 20_000)
        astr1_gd(s, ls)
        @test s.stop_iteration < 20_000
        @test s.grad_val_hist[s.stop_iteration + 1] <= 1e-6
    end

    ## edge cases
    xstar = A \ y
    for variant in (:adagrad, :maxg)
        NLPModels.reset!(ls.counters)
        s = ASTR1GD(T; x0 = copy(xstar), variant = variant, θ = 1.0,
            threshold = 1e-6, max_iterations = 100)
        x = astr1_gd(s, ls)
        @test s.stop_iteration == 0
        @test ls.counters.counters.neval_grad == 1
        @test x == xstar

        NLPModels.reset!(ls.counters)
        s = ASTR1GD(T; x0 = randn(T, 10), variant = variant, θ = 1.0,
            threshold = 0.0, max_iterations = 5)
        astr1_gd(s, ls)
        @test s.stop_iteration == 5
        @test ls.counters.counters.neval_grad == 6
    end
end

end
