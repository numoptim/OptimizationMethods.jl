# Date: 2025/05/08
# Author: Christian Varner
# Purpose: Test cases for the watchdog fixed gd method

module TestWatchdogFixedGD

using Test, OptimizationMethods, LinearAlgebra, CircularArrays, Random

#Random.seed!(1234) # for reproducibility

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

            # test field types
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

        # reset values for the method
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

@testset "Test Method: WatchdogFixedGD{T} Monotone" begin

    # Random parameters
    T = Float64
    dim = 50
    x0 = randn(T, dim)
    α = 1e-5 * rand(T)
    δ = .5
    ρ = 1e-4 * rand(T)
    window_size = 1
    η = rand(T)
    line_search_max_iterations = rand(1:100)
    inner_loop_max_iterations = rand(1:100)
    threshold = rand(T)
    max_iterations = rand(10:25) 

    # first inner loop fails -- line search succeeds
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        η = η, line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedGD(
            T;
            x0 = x0,
            α = 10.0,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = 100,
            window_size = window_size, 
            η = η,
            inner_loop_max_iterations = 1,
            threshold = threshold,
            max_iterations = 1
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x = watchdog_fixed_gd(optData, progData)

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
        @test optData.F_θk == F(x0)
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
        η = η, line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedGD(
            T;
            x0 = x0,
            α = 10.0,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = 0,
            window_size = window_size, 
            η = η,
            inner_loop_max_iterations = 1,
            threshold = threshold,
            max_iterations = 1
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x = watchdog_fixed_gd(optData, progData)

        # check that we return x0
        @test x ≈ x0

        # check stop iteration
        @test optData.stop_iteration == 0
    end

    # first inner loop succeeds
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        η = η, line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedGD(
            T;
            x0 = x0,
            α = 1e-10,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = line_search_max_iterations,
            window_size = window_size, 
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = 1
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x1 = watchdog_fixed_gd(optData, progData)

        optData_0 = WatchdogFixedGD(
            T;
            x0 = x0,
            α = 1e-10,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = line_search_max_iterations,
            window_size = window_size, 
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = 0
        )
        x_0 = watchdog_fixed_gd(optData_0, progData) 

        # set up for the inner loop
        precomp, store = OptimizationMethods.initialize(progData)
        F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
        G(θ) = OptimizationMethods.grad!(progData, precomp, store, θ)
        optData_0.F_θk = F(x_0)

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
        η = η, line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
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

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        xk = watchdog_fixed_gd(optData, progData)

        optData_km1 = WatchdogFixedGD(
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
            max_iterations = max_iterations - 1
        )
        xkm1 = watchdog_fixed_gd(optData_km1, progData) 

        # set up for the inner loop
        precomp, store = OptimizationMethods.initialize(progData)
        F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
        G(θ) = OptimizationMethods.grad!(progData, precomp, store, θ)
        optData_km1.F_θk = F(xkm1)

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

@testset "Test Method: WatchdogFixedGD{T} Nonmonotone" begin

    # Random parameters
    T = Float64
    dim = 50
    x0 = randn(T, dim)
    α = 1e-5 * rand(T)
    δ = 0.5
    ρ = 1e-4 * rand(T)
    window_size = rand(2:10)
    η = rand(T)
    line_search_max_iterations = rand(1:100)
    inner_loop_max_iterations = rand(1:100)
    threshold = rand(T)
    max_iterations = rand(window_size:25) 

    # first inner loop fails
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        η = η, line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedGD(
            T;
            x0 = x0,
            α = 10.0,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = 100,
            window_size = window_size, 
            η = η,
            inner_loop_max_iterations = 1,
            threshold = threshold,
            max_iterations = 1
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x = watchdog_fixed_gd(optData, progData)

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
        @test optData.F_θk == F(x0)
        @test optData.∇F_θk ≈ g0

        # check histories
        g1 = OptimizationMethods.grad(progData, x1)
        @test optData.grad_val_hist[2] ≈ norm(g1)
        @test optData.iter_hist[2] ≈ x1

        # check objective hist
        @test optData.objective_hist[2] ≈ F(x1)
        @test optData.reference_value ≈ F(x0)
        @test optData.reference_value_index ≈ 1

        # check stop iteration
        @test optData.stop_iteration == 1
    end

    # first inner loop fails -- line search fails
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        η = η, line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedGD(
            T;
            x0 = x0,
            α = 10.0,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = 0,
            window_size = window_size, 
            η = η,
            inner_loop_max_iterations = 1,
            threshold = threshold,
            max_iterations = 1
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x = watchdog_fixed_gd(optData, progData)

        # check that we return x0
        @test x ≈ x0

        # check stop iteration
        @test optData.stop_iteration == 0
    end

    # first inner loop succeeds
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        η = η, line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
        # struct
        optData = WatchdogFixedGD(
            T;
            x0 = x0,
            α = 1e-10,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = line_search_max_iterations,
            window_size = window_size, 
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = 1
        )

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x1 = watchdog_fixed_gd(optData, progData)

        optData_0 = WatchdogFixedGD(
            T;
            x0 = x0,
            α = 1e-10,
            δ = δ,
            ρ = ρ, 
            line_search_max_iterations = line_search_max_iterations,
            window_size = window_size, 
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = 0
        )
        x_0 = watchdog_fixed_gd(optData_0, progData) 

        # set up for the inner loop
        precomp, store = OptimizationMethods.initialize(progData)
        F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
        G(θ) = OptimizationMethods.grad!(progData, precomp, store, θ)
        optData_0.F_θk = F(x_0)

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
        @test optData.objective_hist[2] ≈ F(x1)
        @test optData.reference_value ≈ F(x0)
        @test optData.reference_value_index == 1

    end # end of first iteration inner loop succeeds

    # test arbitrary inner loop
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, window_size = window_size,
        η = η, line_search_max_iterations = line_search_max_iterations, 
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold, max_iterations = max_iterations
        
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

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        xk = watchdog_fixed_gd(optData, progData)

        optData_km1 = WatchdogFixedGD(
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
            max_iterations = max_iterations - 1
        )
        xkm1 = watchdog_fixed_gd(optData_km1, progData) 
        τkm1 = optData_km1.reference_value

        # set up for the inner loop
        precomp, store = OptimizationMethods.initialize(progData)
        F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
        G(θ) = OptimizationMethods.grad!(progData, precomp, store, θ)
        optData_km1.F_θk = F(xkm1)

        G(xkm1)
        optData_km1.∇F_θk = store.grad

        # conduct inner loop
        OptimizationMethods.inner_loop!(xkm1, optData_km1.iter_hist[max_iterations], 
            optData_km1, progData, precomp, store, max_iterations; 
            max_iterations = optData_km1.inner_loop_max_iterations)        

        if F(xkm1) <= τkm1 - 
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
                τkm1,
                optData.α,
                optData.δ,
                optData.ρ;
                max_iteration = optData.line_search_max_iterations)
            @test xkm1 ≈ xk
        end

        # test gradient history of optData
        G(xkm1)
        max_val, max_ind = findmax(optData.objective_hist)
        @test optData.grad_val_hist[max_iterations + 1] ≈ norm(store.grad)
        @test optData.objective_hist[optData.reference_value_index] ≈ max_val
        @test optData.reference_value ≈ max_val
    end # end of test for arbitrary inner loop

end # end of Nonmonotone tests

end # end of module