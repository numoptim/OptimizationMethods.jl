# Date: 2025/05/13
# Author: Christian Varner
# Purpose: Test implementation of damped bfgs

module TestWatchdogFixedDampedBFGS

using Test, OptimizationMethods, LinearAlgebra, CircularArrays

@testset "Test WatchdogFixedDampedBFGSGD{T}" begin

    ############################################################################
    # Test the structure definition and fields
    ############################################################################

    # test definition
    @test isdefined(OptimizationMethods, :WatchdogFixedDampedBFGSGD)

    # test unique fields names
    unique_fields = [
        :F_θk,
        :∇F_θk,
        :B_θk,
        :norm_∇F_ψ,
        :c,
        :Bjk,
        :δBjk,
        :rjk,
        :sjk,
        :yjk,
        :d0k,
        :djk,
        :α,
        :δ,
        :ρ,
        :line_search_max_iterations,
        :max_distance_squared,
        :η,
        :inner_loop_max_iterations,
        :objective_hist,
        :reference_value,
        :reference_value_index,
    ]
    let fields = unique_fields
        for field in fields
            @test field in fieldnames(WatchdogFixedDampedBFGSGD)
        end
    end

    # test default field names
    default_fields = [
        :name,
        :threshold,
        :max_iterations,
        :iter_hist,
        :grad_val_hist,
        :stop_iteration,
    ]
    let fields = default_fields
        for field in fields
            @test field in fieldnames(WatchdogFixedDampedBFGSGD)
        end
    end

    # test that I didn't miss any fields
    @test length(unique_fields) + length(default_fields) ==
        length(fieldnames(WatchdogFixedDampedBFGSGD))
    
    ############################################################################
    # Test the error
    ############################################################################
    
    ############################################################################
    # Test the field types
    ############################################################################

    # define field types
    field_types(::Type{T}) where {T} = 
        [
            (:name, String),
            (:F_θk, T),
            (:∇F_θk, Vector{T}),
            (:B_θk, Matrix{T}),
            (:norm_∇F_ψ, T),
            (:c, T),
            (:Bjk, Matrix{T}),
            (:δBjk, Matrix{T}),
            (:rjk, Vector{T}),
            (:sjk, Vector{T}),
            (:yjk, Vector{T}),
            (:d0k, Vector{T}),
            (:djk, Vector{T}),
            (:α, T),
            (:δ, T),
            (:ρ, T),
            (:line_search_max_iterations, Int64),
            (:max_distance_squared, T),
            (:η, T),
            (:inner_loop_max_iterations, Int64),
            (:objective_hist, CircularVector{T, Vector{T}}),
            (:reference_value, T),
            (:reference_value_index, Int64),
            (:threshold, T),
            (:max_iterations, Int64),
            (:iter_hist, Vector{Vector{T}}),
            (:grad_val_hist, Vector{T}),
            (:stop_iteration, Int64),
        ]
    real_types = [Float16, Float32, Float64]

    let field_types = field_types, real_types = real_types
        for T in real_types

            # generate random arguments for constructor
            dim = rand(50:100)
            x0 = randn(T, dim)
            c = rand(T)
            α = rand(T)
            δ = rand(T)
            ρ = rand(T)
            line_search_max_iterations = rand(1:100)
            η = rand(T)
            inner_loop_max_iterations = rand(1:100)
            window_size = rand(1:100)
            threshold = rand(T)
            max_iterations = rand(1:100)

            # construct
            optData = WatchdogFixedDampedBFGSGD(
                T;
                x0 = x0,
                c = c,
                α = α,
                δ = δ,
                ρ = ρ,
                line_search_max_iterations = line_search_max_iterations,
                η = η,
                inner_loop_max_iterations = inner_loop_max_iterations,
                window_size = window_size,
                threshold = threshold,
                max_iterations = max_iterations
            )

            # test fields
            for (field_symbol, field_type) in field_types(T)
                @test field_type == typeof(getfield(optData, field_symbol))
            end

        end
    end
    
    ############################################################################
    # Test the field initializations
    ############################################################################

    let field_types = field_types, real_types = real_types
        for T in real_types

            # generate random arguments for constructor
            dim = rand(50:100)
            x0 = randn(T, dim)
            c = rand(T)
            α = rand(T)
            δ = rand(T)
            ρ = rand(T)
            line_search_max_iterations = rand(1:100)
            η = rand(T)
            inner_loop_max_iterations = rand(1:100)
            window_size = rand(1:100)
            threshold = rand(T)
            max_iterations = rand(1:100)

            # construct
            optData = WatchdogFixedDampedBFGSGD(
                T;
                x0 = x0,
                c = c,
                α = α,
                δ = δ,
                ρ = ρ,
                line_search_max_iterations = line_search_max_iterations,
                η = η,
                inner_loop_max_iterations = inner_loop_max_iterations,
                window_size = window_size,
                threshold = threshold,
                max_iterations = max_iterations
            )

            # test fields
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1

            # BFGS
            @test optData.c == c
            @test size(optData.Bjk) == (dim, dim)
            @test size(optData.δBjk) == (dim, dim)
            @test length(optData.rjk) == dim
            @test length(optData.sjk) == dim
            @test length(optData.yjk) == dim
            @test length(optData.d0k) == dim
            @test length(optData.djk) == dim

            # Line search parameters
            @test optData.α == α
            @test optData.δ == δ
            @test optData.ρ == ρ
            @test optData.line_search_max_iterations == 
                line_search_max_iterations

            # Watchdog stopping parameters
            @test optData.η == η
            @test optData.inner_loop_max_iterations == 
                inner_loop_max_iterations

            # non-monotone line search value
            @test length(optData.objective_hist) == window_size
            @test optData.reference_value == T(0)
            @test optData.reference_value_index == -1
        end
    end

end

@testset "Test WatchdogFixedDampedBFGSGD{T} Inner Loop" begin

    # generate random arguments for constructor
    T = Float64
    dim = rand(50:100)
    x0 = randn(T, dim)
    c = rand(T)
    α = rand(T)
    δ = rand(T)
    ρ = rand(T)
    line_search_max_iterations = rand(1:100)
    η = rand(T)
    inner_loop_max_iterations = rand(1:100)
    window_size = rand(1:100)
    threshold = rand(T)
    max_iterations = rand(1:100)

    # construct
    optData = WatchdogFixedDampedBFGSGD(
        T;
        x0 = x0,
        c = c,
        α = α,
        δ = δ,
        ρ = ρ,
        line_search_max_iterations = line_search_max_iterations,
        η = η,
        inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size,
        threshold = threshold,
        max_iterations = max_iterations
    )

    # problem 
    progData = OptimizationMethods.LogisticRegression(Float64, nvar=dim)
    precomp, store = OptimizationMethods.initialize(progData)

    # test first event: max iterations 
    let optData = optData, progData = progData, precomp = precomp, 
        store = store, ψjk = copy(x0)
        
        k = rand(1:optData.max_iterations)
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 0)

        @test ψjk == x0
        @test j == 0
        @test optData.max_distance_squared == 0
    end

    # test second event: gradient and objective -- hessian works
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)

        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)
        
        # start approximation
        fill!(optData.Bjk, 0)
        OptimizationMethods.add_identity(optData.Bjk,
            optData.c * norm(store.grad))
        Bjk = copy(optData.Bjk)

        # set values to trigger the second event
        optData.η = 1e16                        
        optData.α = 1e-10

        # run inner loop 
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 100)

        # test that the inner loop terminated after one iteration
        @test j == 1

        # test values
        g0 = OptimizationMethods.grad(progData, x0) 
        @test ψjk == x0 - optData.α * optData.d0k
        @test ψjk == x0 - optData.α * optData.djk
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))
        @test optData.d0k ≈ Bjk \ g0
        @test optData.sjk ≈ ψjk - x0
        @test optData.yjk ≈ OptimizationMethods.grad(progData, ψjk) -
            OptimizationMethods.grad(progData, x0)
    end

    # test random iteration
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        optData.α = optData.α
        optData.grad_val_hist[k] = norm(store.grad)
        max_iterations = rand(2:100)

        # Initial approximation
        fill!(optData.Bjk, 0)
        OptimizationMethods.add_identity(optData.Bjk,
            optData.c * norm(store.grad))
        B00 = copy(optData.Bjk)
        g0 = copy(store.grad)


        # run inner loop to get exit iterations
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = max_iterations)

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        OptimizationMethods.hess!(progData, precomp, store, x0)
        optData.α = optData.α
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)

        # Initial approximation
        fill!(optData.Bjk, 0)
        OptimizationMethods.add_identity(optData.Bjk,
            optData.c * norm(store.grad))

        ψjk = copy(x0)

        OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j-1)

        ψ_jm1_k = copy(ψjk)
        maxdist = optData.max_distance_squared
        g_jm1_k = copy(store.grad)
        B_jm1_k = copy(optData.Bjk)
        step = optData.α * (B_jm1_k \ g_jm1_k)

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        OptimizationMethods.hess!(progData, precomp, store, x0)
        optData.α = optData.α
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)

        # Initial approximation
        fill!(optData.Bjk, 0)
        OptimizationMethods.add_identity(optData.Bjk,
            optData.c * norm(store.grad))

        ψjk = copy(x0)

        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j)

        @test ψjk ≈ ψ_jm1_k - step
        @test optData.max_distance_squared == max(norm(ψjk - x0)^2, maxdist)
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))
        @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)

        @test optData.djk ≈ (B_jm1_k \ g_jm1_k)
        @test optData.d0k ≈ B00 \ g0
        @test optData.sjk ≈ ψjk - ψ_jm1_k
        @test optData.yjk ≈ OptimizationMethods.grad(progData, ψjk) -
            OptimizationMethods.grad(progData, ψ_jm1_k) 
    end
end

@testset "Test WatchdogFixedDampedBFGSGD{T} Method Monotone" begin
end

@testset "Test WatchdogFixedDampedBFGSGD{T} Method Nonmonotone" begin
end

end