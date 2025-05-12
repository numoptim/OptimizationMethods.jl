# Date: 2025/05/12
# Author: Christian Varner
# Purpose: Test the modified newton globalized through watchdog

module TestWatchdogFixedModifiedNewton

using Test, OptimizationMethods, Test, LinearAlgebra, CircularArrays

@testset "Test WatchdogFixedMNewtonGD{T}" begin

    ############################################################################
    # Test the structure definition and fields
    ############################################################################

    # test definition
    @test isdefined(OptimizationMethods, :WatchdogFixedMNewtonGD)

    # test field names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist, 
        :grad_val_hist, :stop_iteration]
    let fields = default_fields
        for field in fields
            @test field in fieldnames(WatchdogFixedMNewtonGD)
        end
    end # end of default tests

    # test unique field names
    unique_fields = [:F_θk, :∇F_θk, :norm_∇F_ψ, :β, :λ, 
        :hessian_modification_max_iteration, :d0k, :α, :δ, :ρ,
        :line_search_max_iterations, :max_distance_squared, :η, 
        :inner_loop_max_iterations, :objective_hist, :reference_value,
        :reference_value_index]
    let fields = unique_fields
        for field in fields
            @test field in fieldnames(WatchdogFixedMNewtonGD)
        end
    end # test unique field names 

    # test that I didn't miss anything
    @test length(unique_fields) + length(default_fields) ==
        length(fieldnames(WatchdogFixedMNewtonGD))

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
            (:norm_∇F_ψ, T),
            (:β, T),
            (:λ, T),
            (:hessian_modification_max_iteration, Int64),
            (:d0k, Vector{T}),
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

            # generate random arguments and struct
            dim = rand(50:100)
            x0 = randn(T, dim)
            β = rand(T)
            λ = rand(T)
            hessian_modification_max_iteration = rand(1:100)
            α = rand(T)
            δ = rand(T)
            ρ = rand(T)
            line_search_max_iterations = rand(1:100)
            η = rand(T)
            inner_loop_max_iterations = rand(1:100)
            window_size = rand(1:100)
            threshold = rand(T)
            max_iterations = rand(1:100)

            optData = WatchdogFixedMNewtonGD(T;
                x0 = x0,
                β = β,
                λ = λ, 
                hessian_modification_max_iteration = hessian_modification_max_iteration,
                α = α,
                δ = δ,
                ρ = ρ,
                line_search_max_iterations = line_search_max_iterations,
                η = η,
                inner_loop_max_iterations = inner_loop_max_iterations,
                window_size = window_size,
                threshold = threshold,
                max_iterations = max_iterations)

            # test fields
            for (field_symbol, field_type) in field_types(T)
                @test field_type == typeof(getfield(optData, field_symbol))
            end

        end
    end # test the types of the field
    
    ############################################################################
    # Test the field initializations
    ############################################################################

    let real_types = real_types
        for T in real_types

            # generate random arguments and struct
            dim = rand(50:100)
            x0 = randn(T, dim)
            β = rand(T)
            λ = rand(T)
            hessian_modification_max_iteration = rand(1:100)
            α = rand(T)
            δ = rand(T)
            ρ = rand(T)
            line_search_max_iterations = rand(1:100)
            η = rand(T)
            inner_loop_max_iterations = rand(1:100)
            window_size = rand(1:100)
            threshold = rand(T)
            max_iterations = rand(1:100)

            optData = WatchdogFixedMNewtonGD(T;
                x0 = x0,
                β = β,
                λ = λ, 
                hessian_modification_max_iteration = hessian_modification_max_iteration,
                α = α,
                δ = δ,
                ρ = ρ,
                line_search_max_iterations = line_search_max_iterations,
                η = η,
                inner_loop_max_iterations = inner_loop_max_iterations,
                window_size = window_size,
                threshold = threshold,
                max_iterations = max_iterations)

            # test fields - default
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1

            # test buffer
            @test length(optData.∇F_θk) == dim

            # test modified newton helpers
            @test optData.β == β
            @test optData.λ == λ
            @test optData.hessian_modification_max_iteration == 
                hessian_modification_max_iteration
            @test length(optData.d0k) == dim

            # test line search parameters
            @test optData.α == α
            @test optData.δ == δ
            @test optData.ρ == ρ
            @test optData.line_search_max_iterations == line_search_max_iterations

            # test watchdog stopping parameters
            @test optData.η == η
            @test optData.inner_loop_max_iterations == inner_loop_max_iterations

            # test nonmonotone line search reference values
            @test length(optData.objective_hist) == window_size
            @test optData.reference_value == T(0)
            @test optData.reference_value_index == -1

        end
    end # test the types of the field
end

@testset "Test WatchdogFixedMNewtonGD{T} Inner Loop" begin

    # generate random arguments and struct
    T = Float64
    dim = rand(50:100)
    x0 = randn(T, dim)
    β = rand(T)
    λ = rand(T)
    hessian_modification_max_iteration = rand(1:100)
    α = rand(T)
    δ = rand(T)
    ρ = rand(T)
    line_search_max_iterations = rand(1:100)
    η = rand(T)
    inner_loop_max_iterations = rand(1:100)
    window_size = rand(1:100)
    threshold = rand(T)
    max_iterations = rand(1:100)

    # construct structure
    optData = WatchdogFixedMNewtonGD(T;
        x0 = x0,
        β = β,
        λ = λ, 
        hessian_modification_max_iteration = hessian_modification_max_iteration,
        α = α,
        δ = δ,
        ρ = ρ,
        line_search_max_iterations = line_search_max_iterations,
        η = η,
        inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size,
        threshold = threshold,
        max_iterations = max_iterations)

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
        OptimizationMethods.hess!(progData, precomp, store, x0) 

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
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))

        H0 = OptimizationMethods.hess(progData, x0)
        optData.λ = λ
        res = OptimizationMethods.add_identity_until_pd!(H0;
            λ = optData.λ,
            β = optData.β,
            max_iterations = 
                optData.hessian_modification_max_iteration)
        OptimizationMethods.lower_triangle_solve!(g0, H0')
        OptimizationMethods.upper_triangle_solve!(g0, H0)

        @test res[2]
        @test optData.d0k ≈ g0
    end

    #  # test second event: gradient and objective -- not hessian
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        OptimizationMethods.hess!(progData, precomp, store, x0) 

        # set values to trigger the second event
        optData.η = 1e16                        
        optData.α = 1e-10
        optData.hessian_modification_max_iteration = 0

        # run inner loop 
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 100)

        # test that the inner loop terminated after one iteration
        @test j == 1

        # test values
        g0 = OptimizationMethods.grad(progData, x0) 
        @test ψjk == x0 - optData.α * optData.d0k
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))

        H0 = OptimizationMethods.hess(progData, x0)
        optData.λ = λ
        res = OptimizationMethods.add_identity_until_pd!(H0;
            λ = optData.λ,
            β = optData.β,
            max_iterations = 
                optData.hessian_modification_max_iteration)

        @test !res[2]
        @test optData.d0k ≈ g0
    end

    # test random iteration -- hessian works
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        OptimizationMethods.hess!(progData, precomp, store, x0)
        optData.α = optData.α
        optData.λ = optData.λ
        optData.β = optData.β
        optData.hessian_modification_max_iteration = 100
        optData.grad_val_hist[k] = norm(store.grad)
        max_iterations = rand(2:100)

        # run inner loop to get exit iterations
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = max_iterations)

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        OptimizationMethods.hess!(progData, precomp, store, x0)
        optData.α = optData.α
        optData.λ = optData.λ
        optData.β = optData.β
        optData.grad_val_hist[k] = norm(store.grad)
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)

        ψjk = copy(x0)

        OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j-1)

        ψ_jm1_k = copy(ψjk)
        maxdist = optData.max_distance_squared
        g_jm1_k = copy(store.grad)
        H_jm1_k = OptimizationMethods.hess(progData, ψ_jm1_k)

        optData.λ = λ
        res = OptimizationMethods.add_identity_until_pd!(H_jm1_k;
            λ = optData.λ,
            β = optData.β,
            max_iterations = 
                optData.hessian_modification_max_iteration)
        OptimizationMethods.lower_triangle_solve!(g_jm1_k, H_jm1_k')
        OptimizationMethods.upper_triangle_solve!(g_jm1_k, H_jm1_k)
        step = optData.α * g_jm1_k

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        OptimizationMethods.hess!(progData, precomp, store, x0)
        optData.α = optData.α
        optData.λ = optData.λ
        optData.β = optData.β
        optData.grad_val_hist[k] = norm(store.grad)
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)

        ψjk = copy(x0)

        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j)

        @test ψjk ≈ ψ_jm1_k - step
        @test optData.max_distance_squared == max(norm(ψjk - x0)^2, maxdist)
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))
        @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
        @test store.hess ≈ OptimizationMethods.hess(progData, ψjk)
        @test optData.λ ≈ res[1] / 2

        g_jm1_k = OptimizationMethods.grad(progData, x0)
        H_jm1_k = OptimizationMethods.hess(progData, x0)
        optData.λ = λ
        res = OptimizationMethods.add_identity_until_pd!(H_jm1_k;
            λ = optData.λ,
            β = optData.β,
            max_iterations = 
                optData.hessian_modification_max_iteration)
        OptimizationMethods.lower_triangle_solve!(g_jm1_k, H_jm1_k')
        OptimizationMethods.upper_triangle_solve!(g_jm1_k, H_jm1_k)

        @test g_jm1_k ≈ optData.d0k
    end

    # test random iteration -- hessian fails
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        OptimizationMethods.hess!(progData, precomp, store, x0)
        optData.α = optData.α
        optData.λ = optData.λ
        optData.β = optData.β
        optData.hessian_modification_max_iteration = 0
        optData.grad_val_hist[k] = norm(store.grad)
        max_iterations = rand(2:100)

        # run inner loop to get exit iterations
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = max_iterations)

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        OptimizationMethods.hess!(progData, precomp, store, x0)
        optData.α = optData.α
        optData.λ = optData.λ
        optData.β = optData.β
        optData.grad_val_hist[k] = norm(store.grad)
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)

        ψjk = copy(x0)

        OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j-1)

        ψ_jm1_k = copy(ψjk)
        maxdist = optData.max_distance_squared
        g_jm1_k = copy(store.grad)
        step = optData.α * g_jm1_k

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)       
        OptimizationMethods.hess!(progData, precomp, store, x0)
        optData.α = optData.α
        optData.λ = optData.λ
        optData.β = optData.β
        optData.grad_val_hist[k] = norm(store.grad)
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)

        ψjk = copy(x0)

        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j)

        @test ψjk ≈ ψ_jm1_k - step
        @test optData.max_distance_squared == max(norm(ψjk - x0)^2, maxdist)
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))
        @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
        @test store.hess ≈ OptimizationMethods.hess(progData, ψjk)
        @test optData.d0k ≈ OptimizationMethods.grad(progData, x0) 
    end


end # end of test for inner loop

@testset "Test WatchdogFixedMNewtonGD{T} Method Monotone" begin
end

@testset "Test WatchdogFixedMNewtonGD{T} Method Nonmonotone" begin
end

end