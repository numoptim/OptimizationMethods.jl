module TestGDAdaptiveTrustRegion

using Test, OptimizationMethods, Random, LinearAlgebra, NLPModels

# f(x) = 1/2 x' diag(1, 10) x as an NLPModel-free quadratic via LeastSquares:
# f(x) = 1/2 ||A x - 0||^2 with A = diag(1, sqrt(10)).
quad_problem() = OptimizationMethods.LeastSquares(
    Matrix(Diagonal([1.0, sqrt(10.0)])), zeros(2); x0 = [1.0, 1.0])

@testset "Method: Adaptive Trust Region (ADATR)" begin
    Random.seed!(4242)
    T = Float64

    ## struct
    @test supertype(AdaptiveTrustRegionGD) == OptimizationMethods.AbstractOptimizerData
    for f in (:name, :init_scaling, :α, :b_min, :threshold, :max_iterations,
        :iter_hist, :grad_val_hist, :scaling_hist, :stop_iteration)
        @test f in fieldnames(AdaptiveTrustRegionGD)
    end
    s = AdaptiveTrustRegionGD(T; x0 = zeros(3), init_scaling = 2.0, α = 0.9,
        b_min = 1e-4, threshold = 1e-3, max_iterations = 7)
    @test length(s.iter_hist) == 8 && length(s.grad_val_hist) == 8
    @test length(s.scaling_hist) == 8 && s.stop_iteration == -1
    @test_throws AssertionError AdaptiveTrustRegionGD(T; x0 = zeros(3),
        init_scaling = 1e-5, α = 0.9, b_min = 1e-4, threshold = 1.0, max_iterations = 1)
    @test_throws AssertionError AdaptiveTrustRegionGD(T; x0 = zeros(3),
        init_scaling = 1.0, α = 1.0, b_min = 1e-4, threshold = 1.0, max_iterations = 1)

    ## pure update rule
    # branch 1: ||g|| <= α ω; b̂ = min(b̂max, max(bmin, b/2))
    @test adaptive_trust_region_update(20.0, 10.0, 5.0, 0.9, 1e-4, 10.05) == (10.0, 5.0)
    @test adaptive_trust_region_update(20.0, 10.0, 5.0, 0.9, 1e-4, 4.0) == (4.0, 5.0)
    @test adaptive_trust_region_update(1e-4, 10.0, 5.0, 0.9, 1e-4, 4.0) == (1e-4, 5.0)
    # branch 2
    @test adaptive_trust_region_update(2.0, 10.0, 11.0, 0.9, 1e-4, 4.0) == (2.0 + 121 / 2, 10.0)
    # α = 0: always WNGrad unless gradient is zero
    @test adaptive_trust_region_update(2.0, 10.0, 1e-8, 0.0, 1e-4, 4.0) == (2.0 + 1e-16 / 2, 10.0)

    ## hand-computed iterations on f = 1/2 x' diag(1,10) x, x0 = (1, 1), b0 = 20
    # gradient g(x) = (x1, 10 x2).
    g(x) = [x[1], 10 * x[2]]
    x0 = [1.0, 1.0]
    b0 = 20.0

    # flexible variant α = 0.9, bmin = 1e-4, b̂max = ||g0|| = sqrt(101)
    xa1 = [1 - 1 / 20, 1 - 10 / 20]                     # (0.95, 0.5)
    ga1 = [0.95, 5.0]                                   # ||ga1|| = sqrt(25.9025)
    na1 = sqrt(0.95^2 + 25.0)
    @test na1 <= 0.9 * sqrt(101.0)                       # branch 1
    ba1 = min(sqrt(101.0), max(1e-4, 20 / 2))            # = 10
    xa2 = [0.95 - 0.95 / ba1, 0.5 - 5.0 / ba1]          # (0.855, 0)
    na2 = 0.855
    @test na2 <= 0.9 * na1                               # branch 1 again
    ba2 = min(sqrt(101.0), max(1e-4, ba1 / 2))           # = 5
    xa3 = [0.855 - 0.855 / ba2, 0.0]                    # (0.684, 0)

    prob = quad_problem()
    s = AdaptiveTrustRegionGD(T; x0 = copy(x0), init_scaling = b0, α = 0.9,
        b_min = 1e-4, threshold = 0.0, max_iterations = 3)
    adaptive_trust_region_gd(s, prob)
    @test s.iter_hist[2] ≈ xa1 rtol = 1e-12
    @test s.iter_hist[3] ≈ xa2 rtol = 1e-12 atol = 1e-14
    @test s.iter_hist[4] ≈ xa3 rtol = 1e-12 atol = 1e-14
    @test s.scaling_hist[1:3] ≈ [20.0, ba1, ba2] rtol = 1e-12
    @test s.stop_iteration == 3

    # conservative variant α = 0 (WNGrad): b_{k+1} = b_k + ||g_{k+1}||^2 / b_k
    xc1 = [0.95, 0.5]
    nc1sq = 0.95^2 + 25.0
    bc1 = 20.0 + nc1sq / 20.0
    xc2 = [0.95 - 0.95 / bc1, 0.5 - 5.0 / bc1]
    nc2sq = xc2[1]^2 + 100 * xc2[2]^2
    bc2 = bc1 + nc2sq / bc1
    xc3 = [xc2[1] - xc2[1] / bc2, xc2[2] - 10 * xc2[2] / bc2]

    prob = quad_problem()
    s = AdaptiveTrustRegionGD(T; x0 = copy(x0), init_scaling = b0, α = 0.0,
        b_min = 1e-4, threshold = 0.0, max_iterations = 3)
    adaptive_trust_region_gd(s, prob)
    @test s.iter_hist[2] ≈ xc1 rtol = 1e-12
    @test s.iter_hist[3] ≈ xc2 rtol = 1e-12
    @test s.iter_hist[4] ≈ xc3 rtol = 1e-12
    @test s.scaling_hist[1:3] ≈ [20.0, bc1, bc2] rtol = 1e-12

    ## invariants, objective evaluations, gradient counts
    # Well-conditioned 50x10 least squares, normalized so that the eigenvalues
    # of A'A lie in about [0.3, 2.1] (L ≈ 2). With the default LeastSquares
    # scaling (L ≈ 85, ||∇f(x0)|| ≈ 150) a first step with b0 = 1 overshoots by
    # orders of magnitude and WNGrad's b becomes ~1e7 (known behaviour).
    A = randn(T, 50, 10) / sqrt(50)
    y = A * randn(T, 10) + 0.1 * randn(T, 50)
    ls = OptimizationMethods.LeastSquares(A, y; x0 = zeros(T, 10))
    for α in (0.0, 0.9), b0 in (1e4, 10.0, 1.0, 0.1)
        NLPModels.reset!(ls.counters)
        s = AdaptiveTrustRegionGD(T; x0 = randn(T, 10), init_scaling = b0, α = α,
            b_min = 1e-4, threshold = 1e-6, max_iterations = 2000)
        adaptive_trust_region_gd(s, ls)
        K = s.stop_iteration
        c = ls.counters.counters
        @test c.neval_obj == 0
        @test c.neval_grad == K + 1
        b = s.scaling_hist[1:K+1]
        @test all(b .>= 1e-4)
        for k in 0:K-1
            step = norm(s.iter_hist[k+2] - s.iter_hist[k+1])
            Δk = s.grad_val_hist[k+1] / b[k+1]
            @test isapprox(step, Δk; rtol = 1e-8,
                atol = 1e-12 * norm(s.iter_hist[k+1]))
        end
        if α == 0.0
            # δ_k = 1/b_k non-increasing
            @test all(diff(b) .>= 0)
        end
    end

    ## convergence on a well-conditioned 50x10 least squares, η = 1 (b0 = 1/η)
    for α in (0.0, 0.9)
        NLPModels.reset!(ls.counters)
        s = AdaptiveTrustRegionGD(T; x0 = randn(T, 10), init_scaling = 1.0, α = α,
            b_min = 1e-4, threshold = 1e-6, max_iterations = 20_000)
        adaptive_trust_region_gd(s, ls)
        @test s.stop_iteration < 20_000
        @test s.grad_val_hist[s.stop_iteration + 1] <= 1e-6
    end

    ## edge cases
    xstar = A \ ls.cons
    NLPModels.reset!(ls.counters)
    s = AdaptiveTrustRegionGD(T; x0 = copy(xstar), init_scaling = 1.0, α = 0.9,
        b_min = 1e-4, threshold = 1e-6, max_iterations = 100)
    x = adaptive_trust_region_gd(s, ls)
    @test s.stop_iteration == 0
    @test ls.counters.counters.neval_grad == 1
    @test x == xstar

    NLPModels.reset!(ls.counters)
    s = AdaptiveTrustRegionGD(T; x0 = randn(T, 10), init_scaling = 1.0, α = 0.9,
        b_min = 1e-4, threshold = 0.0, max_iterations = 5)
    adaptive_trust_region_gd(s, ls)
    @test s.stop_iteration == 5
    @test ls.counters.counters.neval_grad == 6
end

end
