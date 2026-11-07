module TestOursVsAlgorithm1

using Test, OptimizationMethods, Random, LinearAlgebra

const OM = OptimizationMethods

# Independent, literal transcription of Algorithm 1 (manuscript, Appendix B of
# the COAP-revision prompt) with the paper's 0-based inner index `j`, and with
# the inner `while` printed the right way round (loop while ‖ψ-θ‖ ≤ 10 AND
# ‖Ḟ(ψ)‖ ∈ (τ_low, τ_upp) AND j < 100). EstimateLipschitz follows the five
# cases documented in `update_local_lipschitz_approximation`, using the paper's
# index: j = 0 has no information (1.0 for k = 0, else the previous estimate);
# j ≥ 1 uses ψ_j, ψ_{j-1}.
#
# `code_terminal = true`: the estimate at the *terminal* inner iterate is
# computed as the code does it, i.e. with the "first iteration" rule when the
# inner loop took exactly one step (j_k = 1). `false`: paper-consistent (the
# estimate is always measured from the last step).
function algorithm1(prob, x0; ρ, δ0, δbar, nouter, code_terminal)
    precomp, store = OM.initialize(prob)
    F(θ) = OM.obj(prob, precomp, store, θ)
    function gradf(θ)
        OM.grad!(prob, precomp, store, θ)
        return copy(store.grad)
    end

    θ = copy(x0); gθ = gradf(θ); Fθ = F(θ)
    τl = norm(gθ) / sqrt(2); τu = sqrt(10) * norm(gθ)
    δ = δ0
    Lcarry = 1.0
    accepted_prev = true
    log = NamedTuple[]
    for k in 0:nouter-1
        ψ = copy(θ); gψ = copy(gθ); j = 0
        L0 = k == 0 ? 1.0 : Lcarry          # L̂_0^k
        L = L0
        α0 = NaN
        while norm(ψ - θ) <= 10 && τl < norm(gψ) < τu && j < 100
            ng = norm(gψ)
            C1 = ng^3 + 0.5 * ng^2 * L + 1e-16
            C2 = ng + 0.5 * L + 1e-16
            α = min(τl^2 / C1, 1 / C2) + 1e-16
            j == 0 && (α0 = α)
            stepnorm = δ * α * ng
            gprev = gψ
            ψ = ψ - (δ * α) * gψ
            gψ = gradf(ψ)
            j += 1
            ratio = norm(gψ - gprev) / stepnorm   # EstimateLipschitz(j, k, ψ_j, ψ_{j-1})
            L = (k == 0 || accepted_prev) ? ratio : max(ratio, L)
        end
        if code_terminal && j == 1
            L = L0
        end
        Lcarry = L
        events = (ball = !(norm(ψ - θ) <= 10), lower = !(τl < norm(gψ)),
            upper = !(norm(gψ) < τu), maxiter = !(j < 100))
        Fψ = F(ψ)
        if Fψ >= Fθ - ρ * δ * α0 * norm(gθ)^2           # reject
            accepted = false
            δ = 0.5 * δ
        elseif norm(gψ) <= τl
            accepted = true
            θ = ψ; gθ = gψ; Fθ = Fψ
            τl = norm(gθ) / sqrt(2); τu = sqrt(20) * τl
        elseif norm(gψ) >= τu
            accepted = true
            θ = ψ; gθ = gψ; Fθ = Fψ
            δ = min(1.5 * δ, δbar)
            τl = norm(gθ) / sqrt(2); τu = sqrt(20) * τl
        else
            accepted = true
            θ = ψ; gθ = gψ; Fθ = Fψ
            δ = min(1.5 * δ, δbar)
        end
        accepted_prev = accepted
        push!(log, (j = j, θ = copy(θ), δ = δ, τl = τl, τu = τu,
            accepted = accepted, events = events))
    end
    return log
end

@testset "OURS (nonsequential_armijo_adaptive_gd) vs Algorithm 1" begin
    Random.seed!(2025)
    T = Float64
    prob = OM.QLLogisticSin(T; nobs = 100, nvar = 10)
    nouter = 40
    for rep in 1:3
        x0 = 20 .* rand(T, 10) .- 10

        # solver
        timed = TimedProblem(prob; tols = [1e-3]); start!(timed)
        s = NonsequentialArmijoAdaptiveGD(T; x0 = copy(x0), δ0 = 1.0, δ_upper = 1.0,
            ρ = 1e-4, threshold = 0.0, max_iterations = nouter)
        nonsequential_armijo_adaptive_gd(s, timed)
        js = [t[2] for t in timed.inner_loops]
        @test [t[1] for t in timed.inner_loops] == collect(1:nouter)

        # transcription reproducing the code's rule at the terminal iterate
        ref = algorithm1(prob, x0; ρ = 1e-4, δ0 = 1.0, δbar = 1.0, nouter = nouter,
            code_terminal = true)
        @test js == [r.j for r in ref]
        for k in 1:nouter
            @test s.iter_hist[k+1] ≈ ref[k].θ rtol = 1e-8 atol = 1e-10
            ev = ref[k].events
            @test s.triggering_event_hist[k][1:4] ==
                  [ev.ball, ev.lower, ev.upper, ev.maxiter]
            @test s.triggering_event_hist[k][5] == ref[k].accepted
        end
        @test s.δk ≈ ref[end].δ rtol = 1e-10
        @test s.τ_lower ≈ ref[end].τl rtol = 1e-8
        @test s.τ_upper ≈ ref[end].τu rtol = 1e-8

        # paper-consistent estimate at the terminal iterate: identical until the
        # loop after the first inner loop with j_k = 1
        ref2 = algorithm1(prob, x0; ρ = 1e-4, δ0 = 1.0, δbar = 1.0, nouter = nouter,
            code_terminal = false)
        k1 = findfirst(==(1), js)
        kmax = k1 === nothing ? nouter : min(nouter, k1)   # loop k1 itself is identical
        for k in 1:kmax
            @test s.iter_hist[k+1] ≈ ref2[k].θ rtol = 1e-8 atol = 1e-10
        end
    end
end

end
