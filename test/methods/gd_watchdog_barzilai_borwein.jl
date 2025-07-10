# Date: 2025/05/09
# Author: Christian Varner
# Purpose: Test implementation

module TestWatchdogBarzilaiBorwein

using Test, OptimizationMethods, LinearAlgebra, CircularArrays

@testset "Test WatchdogSafeBarzilaiBorweinGD{T} Struct" begin

    ############################################################################
    # Test definition and field names
    ############################################################################
    
    # test definition
    @test isdefined(OptimizationMethods, :WatchdogSafeBarzilaiBorweinGD)

    # test unique field names
    unique_fields = [:F_θk, :∇F_θk, :norm_∇F_ψ, :init_stepsize, 
        :bb_step_size, :α_lower, :α_default, :α0k, :iter_diff,
        :grad_diff, :δ, :ρ, :line_search_max_iterations, 
        :max_distance_squared, :η, :inner_loop_max_iterations,
        :objective_hist, :reference_value, :reference_value_index]

    let fields = unique_fields
        for field in fields
            @test field in fieldnames(WatchdogSafeBarzilaiBorweinGD)
        end
    end

    # test default field names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist,
        :grad_val_hist, :stop_iteration]
    
    let fields = default_fields
        for field in fields
            @test field in fieldnames(WatchdogSafeBarzilaiBorweinGD)
        end
    end

    ############################################################################
    # Test errors
    ############################################################################
    
    ############################################################################
    # Test field types
    ############################################################################

    # define the field types
    field_types(::Type{T}) where {T} = [
        (:name, String),
        (:F_θk, T),
        (:∇F_θk, Vector{T}),
        (:norm_∇F_ψ, T),
        (:init_stepsize, T),
        (:α_lower, T),
        (:α_default, T),
        (:α0k, T),
        (:iter_diff, Vector{T}),
        (:grad_diff, Vector{T}),
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

            # define random struct
            dim = 50
            x0 = randn(T, dim)
            init_stepsize = rand(T)
            long_stepsize = rand([false, true])
            α_lower = rand(T)
            α_default = rand(T)
            δ = rand(T)
            ρ = rand(T)
            line_search_max_iterations = rand(1:100)
            η = rand(T)
            inner_loop_max_iterations = rand(1:100)
            window_size = rand(1:100)
            threshold = rand(T)
            max_iterations = rand(1:100)

            # construct
            optData = WatchdogSafeBarzilaiBorweinGD(
                T;
                x0 = x0,
                init_stepsize = init_stepsize,
                long_stepsize = long_stepsize,
                α_lower = α_lower,
                α_default = α_default,
                δ = δ,
                ρ = ρ,
                line_search_max_iterations = line_search_max_iterations,
                η = η,
                inner_loop_max_iterations = inner_loop_max_iterations,
                window_size = window_size,
                threshold = threshold,
                max_iterations = max_iterations
            )

            # test the field types
            for (field_symbol, field_type) in field_types(T)
                @test field_type == typeof(getfield(optData, field_symbol))
            end

        end
    
    end # end of test cases for field types
    
    ############################################################################
    # Test field initializations
    ############################################################################

    real_types = [Float16, Float32, Float64]
    let real_types = real_types

        for T in real_types

            # define random struct
            dim = rand(25:100)
            x0 = randn(T, dim)
            init_stepsize = rand(T)
            long_stepsize = rand([false, true])
            α_lower = rand(T)
            α_default = rand(T)
            δ = rand(T)
            ρ = rand(T)
            line_search_max_iterations = rand(1:100)
            η = rand(T)
            inner_loop_max_iterations = rand(1:100)
            window_size = rand(1:100)
            threshold = rand(T)
            max_iterations = rand(1:100)

            # construct
            optData = WatchdogSafeBarzilaiBorweinGD(
                T;
                x0 = x0,
                init_stepsize = init_stepsize,
                long_stepsize = long_stepsize,
                α_lower = α_lower,
                α_default = α_default,
                δ = δ,
                ρ = ρ,
                line_search_max_iterations = line_search_max_iterations,
                η = η,
                inner_loop_max_iterations = inner_loop_max_iterations,
                window_size = window_size,
                threshold = threshold,
                max_iterations = max_iterations
            )

            # test buffer arrays
            @test length(optData.∇F_θk) == dim

            # test step size helpers
            @test optData.init_stepsize == init_stepsize 
            @test optData.α_lower == α_lower
            @test optData.α_default == α_default
            @test length(optData.iter_diff) == dim 
            @test length(optData.grad_diff) == dim

            # test line search parameters
            @test optData.δ == δ
            @test optData.ρ == ρ
            @test optData.line_search_max_iterations == 
                line_search_max_iterations

            # test watchdog stopping parameters
            @test optData.η == η
            @test optData.inner_loop_max_iterations == 
                inner_loop_max_iterations

            # test nonmonotone objective cache
            @test length(optData.objective_hist) == window_size

            # test default parameters
            @test optData.threshold == threshold 
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1
        end

    end # end of test for field initializations

end

@testset "Test WatchdogSafeBarzilaiBorweinGD{T} Inner Loop" begin

    # define random struct
    T = Float64 
    dim = rand(25:100)
    x0 = randn(T, dim)
    init_stepsize = rand(T)
    long_stepsize = rand([false, true])
    α_lower = rand(T)
    α_default = rand(T)
    δ = rand(T)
    ρ = rand(T)
    line_search_max_iterations = rand(1:100)
    η = rand(T)
    inner_loop_max_iterations = rand(1:100)
    window_size = rand(1:100)
    threshold = rand(T)
    max_iterations = rand(5:25)

    # construct
    optData = WatchdogSafeBarzilaiBorweinGD(
        T;
        x0 = x0,
        init_stepsize = init_stepsize,
        long_stepsize = long_stepsize,
        α_lower = α_lower,
        α_default = α_default,
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
    progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)
    precomp, store = OptimizationMethods.initialize(progData)

    # test first event: max iterations
    let optData = optData, progData = progData, precomp = precomp, 
        store = store, ψjk = copy(x0)
        
        k = rand(1:optData.max_iterations)
        optData.iter_diff = rand(dim)
        optData.grad_diff = rand(dim)
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 0)

        @test ψjk == x0
        @test j == 0
        @test optData.max_distance_squared == 0
    end

    # test second event: gradient and gradient k == 1
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = 1

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = 1e16
        OptimizationMethods.grad!(progData, precomp, store, x0)    
        optData.iter_diff = rand(dim)
        optData.grad_diff = rand(dim)

        # compute step size for testing
        step_size = k == 1 ? optData.init_stepsize : 
        optData.bb_step_size(optData.iter_diff, optData.grad_diff)
        if step_size < optData.α_lower || step_size > (1/optData.α_lower)
            step_size = optData.α_default
        end 

        # set values to trigger the second event
        optData.reference_value = 1e16
        optData.η = 1e16                        

        # run inner loop 
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 100)

        # test that the inner loop terminated after one iteration
        @test j == 1

        # test values
        g0 = OptimizationMethods.grad(progData, x0) 
        g1 = OptimizationMethods.grad(progData, ψjk)  

        @test ψjk ≈ x0 - optData.α0k * g0
        @test optData.α0k ≈ step_size
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))
        @test optData.iter_diff ≈ ψjk - x0
        @test optData.grad_diff ≈ g1 - g0
    end

    # test second event: gradient and gradient k > 1
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(2:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = 1e16
        OptimizationMethods.grad!(progData, precomp, store, x0)    
        optData.iter_diff = rand(dim)
        optData.grad_diff = rand(dim)

        # compute step size for testing
        step_size = k == 1 ? optData.init_stepsize : 
        optData.bb_step_size(optData.iter_diff, optData.grad_diff)
        if step_size < optData.α_lower || step_size > (1/optData.α_lower)
            step_size = optData.α_default
        end 

        # set values to trigger the second event
        optData.reference_value = 1e16
        optData.η = 1e16                        

        # run inner loop 
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 100)

        # test that the inner loop terminated after one iteration
        @test j == 1

        # test values
        g0 = OptimizationMethods.grad(progData, x0) 
        g1 = OptimizationMethods.grad(progData, ψjk)  

        @test ψjk ≈ x0 - optData.α0k * g0
        @test optData.α0k ≈ step_size
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))
        @test optData.iter_diff ≈ ψjk - x0
        @test optData.grad_diff ≈ g1 - g0
    end

    # test first iteration k == 1
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = 1

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        
        optData.η = η       
        optData.iter_diff = rand(dim)
        optData.grad_diff = rand(dim)                 

        # compute step size for testing
        step_size = k == 1 ? optData.init_stepsize : 
        optData.bb_step_size(optData.iter_diff, optData.grad_diff)
        if step_size < optData.α_lower || step_size > (1/optData.α_lower)
            step_size = optData.α_default
        end 

        # run inner loop 
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 1)

        # compute values for test cases
        g0 = OptimizationMethods.grad(progData, x0) 
        g1 = OptimizationMethods.grad(progData, ψjk)  

        @test ψjk ≈ x0 - optData.α0k * g0
        @test optData.α0k ≈ step_size
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))
        @test optData.iter_diff ≈ ψjk - x0
        @test optData.grad_diff ≈ g1 - g0
    end

    # test first iteration k > 1
    let optData = optData, progData = progData, precomp = precomp,
        store = store, ψjk = copy(x0)

        k = rand(1:optData.max_iterations)

        # reset values for the methd
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        
        optData.η = η       
        optData.iter_diff = rand(dim)
        optData.grad_diff = rand(dim)                 

        # compute step size for testing
        step_size = k == 1 ? optData.init_stepsize : 
        optData.bb_step_size(optData.iter_diff, optData.grad_diff)
        if step_size < optData.α_lower || step_size > (1/optData.α_lower)
            step_size = optData.α_default
        end 

        # run inner loop 
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = 1)

        # compute values for test cases
        g0 = OptimizationMethods.grad(progData, x0) 
        g1 = OptimizationMethods.grad(progData, ψjk)  

        @test ψjk ≈ x0 - optData.α0k * g0
        @test optData.α0k ≈ step_size
        @test optData.max_distance_squared == norm(ψjk - x0)^2
        @test optData.norm_∇F_ψ ≈
            norm(OptimizationMethods.grad(progData, ψjk))
        @test optData.iter_diff ≈ ψjk - x0
        @test optData.grad_diff ≈ g1 - g0
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
        
        iter_diff = rand(dim)
        grad_diff = rand(dim)
        optData.iter_diff = iter_diff
        optData.grad_diff = grad_diff
        max_iterations = rand(2:100)
        optData.grad_val_hist[k] = norm(store.grad)

        # run inner loop to get exit iterations
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = max_iterations)

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        
        optData.η = η    
        optData.iter_diff = iter_diff
        optData.grad_diff = grad_diff
        max_iterations = rand(2:100)
        optData.grad_val_hist[k] = norm(store.grad)

        ψjk = copy(x0)

        OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j-1)

        # save required values
        ψ_jm1_k = copy(ψjk)
        maxdist = optData.max_distance_squared
        grd = copy(store.grad)

        # compute step size and step
        step_size = k == 1 ? optData.init_stepsize : 
        optData.bb_step_size(optData.iter_diff, optData.grad_diff)
        if step_size < optData.α_lower || step_size > (1/optData.α_lower) || isnan(step_size)
            step_size = optData.α_default
        end 
        step = step_size * grd

        # reset
        optData.F_θk = OptimizationMethods.obj(progData, x0)
        optData.reference_value = OptimizationMethods.obj(progData, x0)
        OptimizationMethods.grad!(progData, precomp, store, x0)        
        optData.η = η    
        optData.iter_diff = iter_diff
        optData.grad_diff = grad_diff
        max_iterations = rand(2:100)
        optData.grad_val_hist[k] = norm(store.grad)

        ψjk = copy(x0)

        # run inner loop right
        j = OptimizationMethods.inner_loop!(ψjk, x0, optData, progData, precomp,
            store, k; max_iterations = j)

        # check values
        gjk = OptimizationMethods.grad(progData, ψjk)
        gjm1k = OptimizationMethods.grad(progData, ψ_jm1_k)
        @test ψjk ≈ ψ_jm1_k - step
        @test optData.max_distance_squared == max(norm(ψjk - x0)^2, maxdist)
        @test optData.norm_∇F_ψ ≈ norm(gjk)
        @test store.grad ≈ gjk
        @test optData.iter_diff ≈ ψjk - ψ_jm1_k
        @test optData.grad_diff ≈ gjk - gjm1k
    end

end

@testset "Test WatchdogSafeBarzilaiBorweinGD{T} Monotone" begin

    # define random struct
    T = Float64 
    dim = rand(25:100)
    x0 = randn(T, dim)
    init_stepsize = rand(T)
    long_stepsize = rand([false, true])
    α_lower = 1e-10*rand(T)
    α_default = rand(T)
    δ = 0.5
    ρ = 1e-4 * rand(T)
    line_search_max_iterations = rand(50:100)
    η = rand(T)
    inner_loop_max_iterations = rand(1:100)
    window_size = 1
    threshold = rand(T)
    max_iterations = rand(2:10)

    # first inner loop fails -- line search succeeds
    let dim = dim, x0 = x0, init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, α_lower = α_lower,
        α_default = α_default, δ = δ, ρ = ρ, 
        line_search_max_iterations = line_search_max_iterations,
        η = η, inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        optData = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1.,
            α_default = 10.0,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = 100,
            η = η,
            inner_loop_max_iterations = 1,
            window_size = window_size,
            threshold = threshold,
            max_iterations = 1)        

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x = watchdog_safe_barzilai_borwein_gd(optData, progData)

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
                optData.α0k,
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
    let dim = dim, x0 = x0, init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, α_lower = α_lower,
        α_default = α_default, δ = δ, ρ = ρ, 
        line_search_max_iterations = line_search_max_iterations,
        η = η, inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        optData = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1.,
            α_default = 10.0,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = 0,
            η = η,
            inner_loop_max_iterations = 1,
            window_size = window_size,
            threshold = threshold,
            max_iterations = 1)        

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x = watchdog_safe_barzilai_borwein_gd(optData, progData)

        # check that we return x0
        @test x ≈ x0

        # check stop iteration
        @test optData.stop_iteration == 0
    end

    # first inner loop succeeds
    let dim = dim, x0 = x0, init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, α_lower = α_lower,
        α_default = α_default, δ = δ, ρ = ρ, 
        line_search_max_iterations = line_search_max_iterations,
        η = η, inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        optData = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1.0,
            α_default = 1e-10,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = line_search_max_iterations,
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            window_size = window_size,
            threshold = threshold,
            max_iterations = 1)        

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x1 = watchdog_safe_barzilai_borwein_gd(optData, progData)

        optData_0 = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1.0,
            α_default = 1e-10,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = line_search_max_iterations,
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            window_size = window_size,
            threshold = threshold,
            max_iterations = 0) 
        x_0 = watchdog_safe_barzilai_borwein_gd(optData_0, progData) 

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
    let dim = dim, x0 = x0, init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, α_lower = α_lower,
        α_default = α_default, δ = δ, ρ = ρ, 
        line_search_max_iterations = line_search_max_iterations,
        η = η, inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        # struct
        optData = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1e-16,
            α_default = α_default,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = line_search_max_iterations,
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            window_size = window_size,
            threshold = 0.0,
            max_iterations = max_iterations)    

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        xk = watchdog_safe_barzilai_borwein_gd(optData, progData)

        optData_km1 = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1e-16,
            α_default = α_default,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = line_search_max_iterations,
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            window_size = window_size,
            threshold = 0.0,
            max_iterations = max_iterations - 1) 
        xkm1 = watchdog_safe_barzilai_borwein_gd(optData_km1, progData) 

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
                optData_km1.α0k,
                optData.δ,
                optData.ρ;
                max_iteration = optData.line_search_max_iterations)
            @test backtrack_success
            @test xkm1 ≈ xk
        end

        # test gradient history of optData
        G(xk)
        @test optData.grad_val_hist[max_iterations + 1] ≈ norm(store.grad)
        @test optData.objective_hist[1] ≈ F(xk)
        @test optData.reference_value ≈ F(xk)
        @test optData.reference_value_index == 1
    end

end

@testset "Test WatchdogSafeBarzilaiBorweinGD{T} Nonmonotone" begin

    # define random struct
    T = Float64 
    dim = rand(25:100)
    x0 = randn(T, dim)
    init_stepsize = rand(T)
    long_stepsize = rand([false, true])
    α_lower = 1e-10*rand(T)
    α_default = rand(T)
    δ = 0.5
    ρ = 1e-4 * rand(T)
    line_search_max_iterations = rand(50:100)
    η = rand(T)
    inner_loop_max_iterations = rand(1:100)
    window_size = rand(2:5)
    threshold = rand(T)
    max_iterations = rand(window_size:20)

    # first inner loop fails -- line search succeeds
    let dim = dim, x0 = x0, init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, α_lower = α_lower,
        α_default = α_default, δ = δ, ρ = ρ, 
        line_search_max_iterations = line_search_max_iterations,
        η = η, inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        optData = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1.,
            α_default = 10.0,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = 100,
            η = η,
            inner_loop_max_iterations = 1,
            window_size = window_size,
            threshold = threshold,
            max_iterations = 1)        

        # get random problem
        progData = OptimizationMethods.LogisticRegression(Float64, nvar=dim)

        # run method
        x = watchdog_safe_barzilai_borwein_gd(optData, progData)

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
                optData.α0k,
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
    let dim = dim, x0 = x0, init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, α_lower = α_lower,
        α_default = α_default, δ = δ, ρ = ρ, 
        line_search_max_iterations = line_search_max_iterations,
        η = η, inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        optData = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1.,
            α_default = 10.0,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = 0,
            η = η,
            inner_loop_max_iterations = 1,
            window_size = window_size,
            threshold = threshold,
            max_iterations = 1)        

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x = watchdog_safe_barzilai_borwein_gd(optData, progData)

        # check that we return x0
        @test x ≈ x0

        # check stop iteration
        @test optData.stop_iteration == 0
    end

    # first inner loop succeeds
    let dim = dim, x0 = x0, init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, α_lower = α_lower,
        α_default = α_default, δ = δ, ρ = ρ, 
        line_search_max_iterations = line_search_max_iterations,
        η = η, inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        optData = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1.0,
            α_default = 1e-10,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = line_search_max_iterations,
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            window_size = window_size,
            threshold = threshold,
            max_iterations = 1)        

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        x1 = watchdog_safe_barzilai_borwein_gd(optData, progData)

        optData_0 = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1.0,
            α_default = 1e-10,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = line_search_max_iterations,
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            window_size = window_size,
            threshold = threshold,
            max_iterations = 0) 
        x_0 = watchdog_safe_barzilai_borwein_gd(optData_0, progData) 

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

    end

    # test arbitrary inner loop
    let dim = dim, x0 = x0, init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, α_lower = α_lower,
        α_default = α_default, δ = δ, ρ = ρ, 
        line_search_max_iterations = line_search_max_iterations,
        η = η, inner_loop_max_iterations = inner_loop_max_iterations,
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        # struct
        optData = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1e-16,
            α_default = α_default,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = line_search_max_iterations,
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            window_size = window_size,
            threshold = 0.0,
            max_iterations = max_iterations)    

        # get random problem
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # run method
        xk = watchdog_safe_barzilai_borwein_gd(optData, progData)

        optData_km1 = WatchdogSafeBarzilaiBorweinGD(
            T;
            x0 = x0,
            init_stepsize = init_stepsize,
            long_stepsize = long_stepsize,
            α_lower = 1e-16,
            α_default = α_default,
            δ = δ,
            ρ = ρ,
            line_search_max_iterations = line_search_max_iterations,
            η = η,
            inner_loop_max_iterations = inner_loop_max_iterations,
            window_size = window_size,
            threshold = 0.0,
            max_iterations = max_iterations - 1) 
        xkm1 = watchdog_safe_barzilai_borwein_gd(optData_km1, progData) 

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
                optData_km1.α0k,
                optData.δ,
                optData.ρ;
                max_iteration = optData.line_search_max_iterations)
            @test backtrack_success
            @test xkm1 ≈ xk
        end

        # test gradient history of optData
        G(xk)
        max_val, max_ind = findmax(optData.objective_hist)
        @test optData.grad_val_hist[max_iterations + 1] ≈ norm(store.grad)
        @test optData.objective_hist[optData.reference_value_index] ≈ max_val
        @test optData.reference_value ≈ max_val
    end

end

end # end test cases