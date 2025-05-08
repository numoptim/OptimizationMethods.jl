# Date: 2025/05/08
# Author: Christian Varner
# Purpose: Test cases for the watchdog fixed gd method

module TestWatchdogFixedGD

using Test, OptimizationMethods, LinearAlgebra, CircularArrays

@testset "Test Structure: WatchdogFixedGD{T}" begin

    ############################################################################
    # Test definition and field names
    ############################################################################
    
    # test definition
    @test isdefined(OptimizationMethods, :WatchdogFixedGD)

    # test default field names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist,
        :grad_val_hist, :stop_iteration]
    let fields = default_fields
        for field in fields
            @test field in fieldnames(WatchdogFixedGD)
        end
    end # end test for default fields

    # test special field names
    unique_fields = [:F_θk, :∇F_θk, :norm_∇F_ψ, :α, :δ, :ρ, 
        :line_search_max_iterations, :max_distance_squared, :η,
        :inner_loop_max_iterations, :objective_hist, :reference_value,
        :reference_value_index]
    let fields = unique_fields
        for field in fields
            @test field in fieldnames(WatchdogFixedGD)
        end
    end # end test for unique fields

    # test that I did not miss any
    @test length(fieldnames(WatchdogFixedGD)) == length(unique_fields) +
        length(default_fields)

    ############################################################################
    # Test errors
    ############################################################################
    
    ############################################################################
    # Test field types
    ############################################################################

    real_types = [Float16, Float32, Float64]
    field_types(::Type{T}) where {T} = 
    [
        (:name, String),
        (:F_θk, T),
        (:∇F_θk, Vector{T}),
        (:norm_∇F_ψ, T),
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

    let field_types = field_types, real_types = real_types, dim = 50
        
        for T in real_types

            # generate a random struct
            x0 = randn(T, dim)
            α = rand(T)
            δ = rand(T)
            ρ = rand(T)
            window_size = rand(1:100)
            η = rand(T)
            line_search_max_iterations = rand(1:100)
            inner_loop_max_iterations = rand(1:100)
            threshold = rand(T)
            max_iterations = rand(1:100)

            # generate struct
            optData = WatchdogFixedGD(
                T;
                x0 = x0,
                α = α,
                δ = δ,
                ρ = ρ, 
                line_search_max_iterations = line_search_max_iterations,
                window_size = window_size, 
                η = η,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations
            )

            # test fielt types
            for (field_symbol, field_type) in field_types(T)
                @test field_type == typeof(getfield(optData, field_symbol))
            end

        end # end test case for field types

    end
    
    ############################################################################
    # Test field initializations
    ############################################################################

    let real_types = real_types, dim = 50
        
        for T in real_types

            # generate a random struct
            x0 = randn(T, dim)
            α = rand(T)
            δ = rand(T)
            ρ = rand(T)
            window_size = rand(1:100)
            η = rand(T)
            line_search_max_iterations = rand(1:100)
            inner_loop_max_iterations = rand(1:100)
            threshold = rand(T)
            max_iterations = rand(1:100)

            # generate struct
            optData = WatchdogFixedGD(
                T;
                x0 = x0,
                α = α,
                δ = δ,
                ρ = ρ, 
                line_search_max_iterations = line_search_max_iterations,
                window_size = window_size, 
                η = η,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations
            )

            # test initial field values
            @test length(optData.∇F_θk) == dim
            
            # line search helpers
            @test optData.α == α 
            @test optData.δ == δ
            @test optData.ρ == ρ
            @test optData.line_search_max_iterations == 
                line_search_max_iterations
            
            # inner loop stopping parameters
            @test optData.η == η
            @test optData.inner_loop_max_iterations == 
                inner_loop_max_iterations

            # nonmonotone line search objective
            @test length(optData.objective_hist) == window_size

            # default parameters
            @test optData.threshold == threshold 
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1

        end # end test case for field types

    end

end

@testset "Test Inner Loop: WatchdogFixedGD{T}" begin

    # generate a random struct
    T = Float64
    dim = 50
    x0 = randn(T, dim)
    α = 1e-5 * rand(T)
    δ = rand(T)
    ρ = rand(T)
    window_size = rand(1:100)
    η = rand(T)
    line_search_max_iterations = rand(1:100)
    inner_loop_max_iterations = rand(1:100)
    threshold = rand(T)
    max_iterations = rand(1:100)

    # struct
    optData = WatchdogFixedGD(
                T;
                x0 = x0,
                α = α,
                δ = δ,
                ρ = ρ, 
                line_search_max_iterations = line_search_max_iterations,
                window_size = window_size, 
                η = η,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations
            )
    
    # problem 
    progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)
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

    # test second event: gradient and gradient
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        

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
        @test ψjk == x0 - optData.α * g0
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))

    end

    # test first iteration
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        
        optData.η = η                        
        optData.α = α
        k = rand(1:optData.max_iterations)

        # run inner loop 
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 1)

        g0 = OptimizationMethods.grad(progData, x0) 
        @test ψjk ≈ x0 - optData.α * g0
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))
        @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
    end

    # test random iteration
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        
        optData.η = η                        
        optData.α = α
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)
        max_iterations = rand(2:100)

        # run inner loop to get exit iterations
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = max_iterations)

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        
        optData.η = η                        
        optData.α = α
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)

        ψjk = copy(x0)

        OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j-1)

        ψ_jm1_k = copy(ψjk)
        maxdist = optData.max_distance_squared
        grd = copy(store.grad)
        step = optData.α * grd

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        
        optData.η = η                        
        optData.α = α
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
    end

end

@testset "Test Method: WatchdogFixedGD{T}" begin
end

end