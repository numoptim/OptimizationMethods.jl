# Date: 2025/22/10
# Author: Christian Varner
# Purpose: Tests for gd_fixed_watchdog_gd_simple.jl

module TestWatchdogFixedGDSimple

using Test, OptimizationMethods, LinearAlgebra, CircularArrays, Random

@testset "Test Structure -- WatchdogFixedSimpleGD{T}" begin

    ############################################################################
    # Test definition and field names
    ############################################################################
    
    # test definition
    @test isdefined(OptimizationMethods, :WatchdogFixedSimpleGD)

    # test default field names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist,
        :grad_val_hist, :stop_iteration]
    let fields = default_fields
        for field in fields
            @test field in fieldnames(WatchdogFixedSimpleGD)
        end
    end # end test for default fields

    # test special field names
    unique_fields = [:∇F_θk, :α, :δ, :ρ, 
        :line_search_max_iterations, :max_distance_squared,
        :inner_loop_max_iterations, :objective_hist, :reference_value,
        :reference_value_index]
    let fields = unique_fields
        for field in fields
            @test field in fieldnames(WatchdogFixedSimpleGD)
        end
    end # end test for unique fields

    # test that I did not miss any
    @test length(fieldnames(WatchdogFixedSimpleGD)) == length(unique_fields) +
        length(default_fields)

    ############################################################################
    # Test field types
    ############################################################################

    real_types = [Float16, Float32, Float64]
    field_types(::Type{T}) where {T} = 
    [
        (:name, String),
        (:∇F_θk, Vector{T}),
        (:α, T),
        (:δ, T),
        (:ρ, T),
        (:line_search_max_iterations, Int64),
        (:max_distance_squared, T),
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
            line_search_max_iterations = rand(1:100)
            inner_loop_max_iterations = rand(1:100)
            threshold = rand(T)
            max_iterations = rand(1:100)

            # generate struct
            optData = WatchdogFixedSimpleGD(
                T;
                x0 = x0,
                α = α,
                δ = δ,
                ρ = ρ, 
                line_search_max_iterations = line_search_max_iterations,
                window_size = window_size, 
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations
            )

            # test field types
            for (field_symbol, field_type) in field_types(T)
                @test field_type == typeof(getfield(optData, field_symbol))
            end

        end # end test case for field types
    end # end let block
end # end of test for structure

@testset "Test Inner Loop -- WatchdogFixedSimpleGD{T}" begin

    # generate a random struct
    T = Float64
    dim = 50
    x0 = randn(T, dim)
    α = 1e-5 * rand(T)
    δ = rand(T)
    ρ = rand(T)
    window_size = rand(1:100)
    line_search_max_iterations = rand(1:100)
    inner_loop_max_iterations = rand(1:100)
    threshold = rand(T)
    max_iterations = rand(1:100)

    # struct
    optData = WatchdogFixedSimpleGD(
                T;
                x0 = x0,
                α = α,
                δ = δ,
                ρ = ρ, 
                line_search_max_iterations = line_search_max_iterations,
                window_size = window_size, 
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

    # test first iteration
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        
        optData.α = α
        k = rand(1:optData.max_iterations)

        # run inner loop 
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 1)

        g0 = OptimizationMethods.grad(progData, x0) 
        @test ψjk ≈ x0 - optData.α * g0
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
    end

    # test random iteration
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)                       
        optData.α = α
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)
        max_iterations = rand(2:100)

        # run inner loop to get exit iterations
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = max_iterations)

        # reset
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)    
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
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)                           
        optData.α = α
        k = rand(1:optData.max_iterations)
        optData.grad_val_hist[k] = norm(store.grad)

        ψjk = copy(x0)

        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j)

        @test ψjk ≈ ψ_jm1_k - step
        @test optData.max_distance_squared == max(norm(ψjk - x0)^2, maxdist)
        @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
    end
end

@testset "Test Full Method -- WatchdogFixedSimpleGD{T}" begin

    # Random parameters
    T = Float64
    dim = 50
    x0 = randn(T, dim)
    α = 1e-5 * rand(T)
    δ = .5
    ρ = 1e-4 * rand(T)
    window_size = 1
    line_search_max_iterations = rand(1:100)
    inner_loop_max_iterations = rand(1:100)
    threshold = rand(T)
    max_iterations = rand(10:25) 

    # first inner loop fails -- line search succeeds
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedSimpleGD(
            T;
            x0 = x0,
            α = 10.0,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = 100,
            window_size = window_size, 
            inner_loop_max_iterations = 1,
            threshold = threshold,
            max_iterations = 1
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x = watchdog_fixed_simple_gd(optData, progData)

        # that x was formed through a backtrack
        x1 = copy(x0)
        F(θ) = OptimizationMethods.obj(progData, θ)
        g0 = OptimizationMethods.grad(progData, x0)
        backtrack_success = OptimizationMethods.backtracking!(
                x1,
                x0,
                F,
                g0,
                norm(g0)^2,
                F(x0),
                optData.α,
                optData.δ,
                optData.ρ;
                max_iteration = optData.line_search_max_iterations)
        
        @test backtrack_success
        @test x1 ≈ x

        # check the θk checkpoints
        @test optData.∇F_θk ≈ g0

        # check histories
        g1 = OptimizationMethods.grad(progData, x1)
        @test optData.grad_val_hist[2] ≈ norm(g1)
        @test optData.iter_hist[2] ≈ x1

        # check objective hist
        @test optData.objective_hist[1] ≈ F(x1)
        @test optData.reference_value ≈ F(x1)
        @test optData.reference_value_index ≈ 1

        # check stop iteration
        @test optData.stop_iteration == 1
    end

    # first inner loop fails -- line search fails
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedSimpleGD(
            T;
            x0 = x0,
            α = 10.0,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = 0,
            window_size = window_size, 
            inner_loop_max_iterations = 1,
            threshold = threshold,
            max_iterations = 1
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x = watchdog_fixed_simple_gd(optData, progData)

        # check that we return x0
        @test x ≈ x0

        # check stop iteration
        @test optData.stop_iteration == 0
    end

    # first inner loop succeeds
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedSimpleGD(
            T;
            x0 = x0,
            α = 1e-10,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = line_search_max_iterations,
            window_size = window_size, 
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = 1
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x1 = watchdog_fixed_simple_gd(optData, progData)

        optData_0 = WatchdogFixedSimpleGD(
            T;
            x0 = x0,
            α = 1e-10,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = line_search_max_iterations,
            window_size = window_size, 
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = 0
        )
        x_0 = watchdog_fixed_simple_gd(optData_0, progData) 

        # set up for the inner loop
        precomp, store = OptimizationMethods.initialize(progData)
        F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
        G(θ) = OptimizationMethods.grad!(progData, precomp, store, θ)

        G(x_0)
        optData_0.∇F_θk = store.grad

        # conduct inner loop
        OptimizationMethods.inner_loop!(x_0, x0, optData_0, progData, 
            precomp, store, 1; 
            max_iterations = optData_0.inner_loop_max_iterations)
        
        
        @test x1 ≈ x_0

        # test gradient history of optData
        G(x1)
        @test optData.grad_val_hist[2] ≈ norm(store.grad)
        @test optData.objective_hist[1] ≈ F(x1)
        @test optData.reference_value ≈ F(x1)
        @test optData.reference_value_index == 1
    end

    # test arbitrary inner loop
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedSimpleGD(
            T;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = line_search_max_iterations,
            window_size = window_size, 
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = max_iterations
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        xk = watchdog_fixed_simple_gd(optData, progData)

        optData_km1 = WatchdogFixedSimpleGD(
            T;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = line_search_max_iterations,
            window_size = window_size, 
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = max_iterations - 1
        )
        xkm1 = watchdog_fixed_simple_gd(optData_km1, progData) 

        # set up for the inner loop
        precomp, store = OptimizationMethods.initialize(progData)
        F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
        G(θ) = OptimizationMethods.grad!(progData, precomp, store, θ)

        G(xkm1)
        optData_km1.∇F_θk = store.grad

        # conduct inner loop
        OptimizationMethods.inner_loop!(xkm1, optData_km1.iter_hist[max_iterations], 
            optData_km1, progData, precomp, store, max_iterations; 
            max_iterations = optData_km1.inner_loop_max_iterations)        

        if F(xkm1) <= F(optData_km1.iter_hist[max_iterations]) - 
                optData_km1.ρ * optData_km1.max_distance_squared
            @test xkm1 ≈ xk
        else
            xkm1 = copy(optData_km1.iter_hist[max_iterations])
            backtrack_success = OptimizationMethods.backtracking!(
                xkm1,
                optData_km1.iter_hist[max_iterations],
                F,
                optData_km1.∇F_θk,
                norm(optData_km1.∇F_θk)^2,
                F(xkm1),
                optData.α,
                optData.δ,
                optData.ρ;
                max_iteration = optData.line_search_max_iterations)
            @test xkm1 ≈ xk
        end

        # test gradient history of optData
        G(xkm1)
        @test optData.grad_val_hist[max_iterations + 1] ≈ norm(store.grad)
        @test optData.objective_hist[1] ≈ F(xk)
        @test optData.reference_value ≈ F(xk)
        @test optData.reference_value_index == 1
    end 

end

end # end of module