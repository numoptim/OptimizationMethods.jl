# Date: 2025/04/21
# Author: Christian Varner
# Purpose: Test the implementation of (nonmonotone) backtracking 
# with modified newton method

module TestModifiedNewtonStepFixStepSizeNLSMaxVal

using Test, OptimizationMethods, CircularArrays, LinearAlgebra

@testset "Test -- FixedModifiedNewtonNLSMaxValGD Structure" begin

    ############################################################################
    # Test the definition of structure
    ############################################################################
    
    @test isdefined(OptimizationMethods, :FixedModifiedNewtonNLSMaxValGD)

    ############################################################################
    # Test the field names in the structure
    ############################################################################

    # test the default field names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist,
        :grad_val_hist, :stop_iteration]
    let fields = default_fields 
    
        # test that the field exists
        for field in fields
            @test field in fieldnames(FixedModifiedNewtonNLSMaxValGD)
        end
    
    end
    
    # test the unique field names
    unique_fields = [:α, :δ, :ρ, :line_search_max_iteration, :step,
        :window_size, :objective_hist, :max_value, :max_index, :β,
        :λ, :hessian_modification_max_iteration]
    let fields = unique_fields 
    
        # test that the field exists
        for field in fields
            @test field in fieldnames(FixedModifiedNewtonNLSMaxValGD)
        end
    
    end

    # test to make sure we didn't miss any fields
    @test length(fieldnames(FixedModifiedNewtonNLSMaxValGD)) == 
        length(default_fields) + length(unique_fields)

    ############################################################################
    # Test the constructor -- errors
    ############################################################################



    ############################################################################
    # Test the constructor -- field types
    ############################################################################
    
    real_types = [Float16, Float32, Float64]
    field_types(type::T) where {T} =
        [
            (:name, String),
            (:α, type),
            (:δ, type),
            (:ρ, type),
            (:line_search_max_iteration, Int64),
            (:step, Vector{type}),
            (:window_size, Int64),
            (:objective_hist, CircularVector{type, Vector{type}}),
            (:max_value, type),
            (:max_index, Int64),
            (:β, type),
            (:λ, type),
            (:hessian_modification_max_iteration, Int64),
            (:threshold, type),
            (:max_iterations, Int64),
            (:iter_hist, Vector{Vector{type}}),
            (:grad_val_hist, Vector{type}),
            (:stop_iteration, Int64)
        ]
       
    # test case for field types    
    let real_types = real_types, field_types = field_types
        
        for type in real_types
            
            # sample random values for the constructor
            dim = 50
            x0 = randn(type, dim) 
            α = rand(type)
            δ = rand(type)
            ρ = rand(type) 
            line_search_max_iteration = rand(1:100)
            window_size = rand(1:100)
            β = rand(type)
            λ = rand(type)
            hessian_modification_max_iteration = rand(1:100)
            threshold = rand(type)
            max_iterations = rand(1:100)

            # get initialized struct
            optData = FixedModifiedNewtonNLSMaxValGD(type;
                x0 = x0,
                α = α,
                δ = δ,
                ρ = ρ,
                line_search_max_iteration = line_search_max_iteration,
                window_size = window_size,
                β = β,
                λ = λ,
                hessian_modification_max_iteration = hessian_modification_max_iteration,
                threshold = threshold,
                max_iterations = max_iterations)

            # test values
            for (field_symbol, field_type) in field_types(type)
                @test field_type == typeof(getfield(optData, field_symbol))
            end

        end

    end # finish the test cases for the field types

    ############################################################################
    # Test the constructor -- initial values
    ############################################################################

    # test the initial values 
    let real_types = real_types
        
        for type in real_types
            
            # sample random values for the constructor
            dim = 50
            x0 = randn(type, dim) 
            α = rand(type)
            δ = rand(type)
            ρ = rand(type) 
            line_search_max_iteration = rand(1:100)
            window_size = rand(1:100)
            β = rand(type)
            λ = rand(type)
            threshold = rand(type)
            hessian_modification_max_iteration = rand(1:100)
            max_iterations = rand(1:100)

            # get initialized struct
            optData = FixedModifiedNewtonNLSMaxValGD(type;
                x0 = x0,
                α = α,
                δ = δ,
                ρ = ρ,
                line_search_max_iteration = line_search_max_iteration,
                window_size = window_size,
                β = β,
                λ = λ,
                hessian_modification_max_iteration = hessian_modification_max_iteration,
                threshold = threshold,
                max_iterations = max_iterations)

            # test line search parameters
            @test optData.α == α
            @test optData.δ == δ
            @test optData.ρ == ρ
            @test optData.line_search_max_iteration == line_search_max_iteration
            @test length(optData.step) == dim

            # test the non-monotone objective history parameters
            @test optData.window_size == window_size
            @test length(optData.objective_hist) == window_size
            @test optData.max_value == type(0)
            @test optData.max_index == -1

            # test the parameters for modified newton
            @test optData.β == β
            @test optData.λ == λ
            @test optData.hessian_modification_max_iteration == 
                hessian_modification_max_iteration

            # test default parameters
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1
        end

    end # end the test cases for the initial values

end # End structure test cases

@testset "Test -- fixed_modified_newton_nls_maxval_gd (Monotone)" begin

    # initialize a random logistic regression problem for testing
    progData = OptimizationMethods.LogisticRegression(Float64)

    # sample random fields for initialization
    dim = 50
    x0 = randn(dim) 
    α = rand()
    δ = rand()
    ρ = 1e-4 
    line_search_max_iteration = rand(50:100)
    window_size = 1
    β = rand()
    λ = rand()
    hessian_modification_max_iteration = rand(10:100)
    threshold = 1e-4

    ############################################################################
    # Line search should be successful
    ############################################################################

    # Base case: test the first iteration of the method
    max_iterations = 1
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, 
        hessian_modification_max_iteration = hessian_modification_max_iteration,
        β = β, λ = λ, threshold = threshold, 
        max_iterations = max_iterations

        # initialize the optimization data
        optData = FixedModifiedNewtonNLSMaxValGD(Float64;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            β = β,
            λ = λ,
            hessian_modification_max_iteration = hessian_modification_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations)

        # get the output after one iteration of the method
        x1 = fixed_modified_newton_nls_maxval_gd(optData, progData)

        # check iterate histories
        @test 1 == optData.stop_iteration
        
        @test x0 == optData.iter_hist[1]
        @test norm(OptimizationMethods.grad(progData, x0)) == 
            optData.grad_val_hist[1]

        @test x1 == optData.iter_hist[2]
        @test norm(OptimizationMethods.grad(progData, x1)) ==
            optData.grad_val_hist[2]

        # get the step
        g0 = OptimizationMethods.grad(progData, x0)
        H0 = OptimizationMethods.hess(progData, x0)
        res = OptimizationMethods.add_identity_until_pd!(
            H0; 
            λ = λ, 
            β = optData.β, 
            max_iterations = optData.hessian_modification_max_iteration
        )
        step = copy(g0)
        if res[2]
            OptimizationMethods.lower_triangle_solve!(step, H0')
            OptimizationMethods.upper_triangle_solve!(step, H0) 
        end
        @test res[2]
        @test optData.λ == res[1] / 2
        @test optData.step ≈ step

        # check backtracing was used
        x = copy(x0)
        F(θ) = OptimizationMethods.obj(progData, θ)
        success = OptimizationMethods.backtracking!(
            x, 
            optData.iter_hist[1],
            F,
            g0,
            step,
            F(x0),
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration)
        @test success
        @test x ≈ x1

        # check the objective history
        @test optData.objective_hist[1] == F(x1)
        @test optData.max_value == F(x1)
        @test optData.max_index == 1
        @test optData.window_size == 1
        @test length(optData.objective_hist) == 1
    end # end first iteration test

    # Inductive step: test a random iteration of the method
    max_iterations = rand(2:10)
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, 
        hessian_modification_max_iteration = hessian_modification_max_iteration,
        β = β, λ = λ, threshold = 1e-10, 
        max_iterations = max_iterations
    
        # initialize the optimization data for max_iteration - 1
        optData = FixedModifiedNewtonNLSMaxValGD(Float64;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            β = β,
            λ = λ,
            hessian_modification_max_iteration = hessian_modification_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations - 1)

        # get the k-1th iteration of the method and λkm1
        xkm1 = fixed_modified_newton_nls_maxval_gd(optData, progData)
        λkm1 = optData.λ

        # initialize the optimization data for max_iteration
        optData = FixedModifiedNewtonNLSMaxValGD(Float64;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            β = β,
            λ = λ,
            hessian_modification_max_iteration = hessian_modification_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations)
        
        # get the kth iteration of the method
        xk = fixed_modified_newton_nls_maxval_gd(optData, progData)

        # check the iterate and gradient histories
        k = optData.stop_iteration

        @test xkm1 == optData.iter_hist[k]
        @test norm(OptimizationMethods.grad(progData, xkm1)) ==
            optData.grad_val_hist[k]

        @test xk == optData.iter_hist[k + 1]
        @test norm(OptimizationMethods.grad(progData, xk)) ==
            optData.grad_val_hist[k + 1]

        # obtain the step taken d_k-1
        gkm1 = OptimizationMethods.grad(progData, xkm1)
        Hkm1 = OptimizationMethods.hess(progData, xkm1)
        res = OptimizationMethods.add_identity_until_pd!(
            Hkm1; 
            λ = λkm1, 
            β = optData.β, 
            max_iterations = optData.hessian_modification_max_iteration
        )
        step = copy(gkm1)
        if res[2]
            OptimizationMethods.lower_triangle_solve!(step, Hkm1')
            OptimizationMethods.upper_triangle_solve!(step, Hkm1) 
        end
        @test res[2]
        @test optData.λ == res[1] / 2
        @test optData.step ≈ step

        # check that backtracking was used
        x = copy(xkm1)
        F(θ) = OptimizationMethods.obj(progData, θ)
        success = OptimizationMethods.backtracking!(
            x,
            xkm1, 
            F,
            gkm1,
            step,
            F(xkm1),
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration
        )
        @test success
        @test x ≈ xk

        # check the objective history
        @test optData.objective_hist[1] == F(xk)
        @test optData.max_value == F(xk)
        @test optData.max_index == 1
    end # end arbitrary iteration test

    ############################################################################
    # Line search failure on first iteration
    ############################################################################

    # Check to make sure the correct iterate is being returned
    line_search_max_iteration = 0
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, 
        hessian_modification_max_iteration = hessian_modification_max_iteration,
        β = β, λ = λ, threshold = 1e-10, 
        max_iterations = max_iterations
    
        # initialize the optimization data
        optData = FixedModifiedNewtonNLSMaxValGD(Float64;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            β = β,
            λ = λ,
            hessian_modification_max_iteration = hessian_modification_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations)

        # get first iteration
        x1 = fixed_modified_newton_nls_maxval_gd(optData, progData)

        # check that correct iterate was returned
        @test optData.stop_iteration == 0
        @test x1 == x0
        
        # test histories
        @test optData.iter_hist[1] == x0
        @test optData.grad_val_hist[1] == norm(OptimizationMethods.grad(progData, x0))
        
        # check the objective histories
        @test optData.max_value == OptimizationMethods.obj(progData, x0)
        @test optData.max_index == 1
        @test optData.objective_hist[1] == OptimizationMethods.obj(progData, x0)
    end # end first iteration line search failer test
end # end of monotone method implementation test cases

@testset "Test - fixed_modified_newton_nls_maxval_gd (Nonmonotone)" begin

    # initialize a random logistic regression problem for testing
    progData = OptimizationMethods.LogisticRegression(Float64)

    # sample random fields for initialization
    dim = 50
    x0 = randn(dim) 
    α = rand()
    δ = rand()
    ρ = 1e-4 
    line_search_max_iteration = rand(50:100)
    window_size = rand(2:10)
    β = rand()
    λ = rand()
    hessian_modification_max_iteration = rand(10:100)
    threshold = 1e-10

    ############################################################################
    # Line search should be successful
    ############################################################################

    # Base case: test the first iteration of the method
    max_iterations = 1
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, 
        hessian_modification_max_iteration = hessian_modification_max_iteration,
        β = β, λ = λ, threshold = threshold, 
        max_iterations = max_iterations

        # initialize the optimization data
        optData = FixedModifiedNewtonNLSMaxValGD(Float64;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            β = β,
            λ = λ,
            hessian_modification_max_iteration = hessian_modification_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations)

        # get the output after one iteration of the method
        x1 = fixed_modified_newton_nls_maxval_gd(optData, progData)

        # check iterate histories
        @test 1 == optData.stop_iteration
        
        @test x0 == optData.iter_hist[1]
        @test norm(OptimizationMethods.grad(progData, x0)) == 
            optData.grad_val_hist[1]

        @test x1 == optData.iter_hist[2]
        @test norm(OptimizationMethods.grad(progData, x1)) ==
            optData.grad_val_hist[2]

        # get the step
        g0 = OptimizationMethods.grad(progData, x0)
        H0 = OptimizationMethods.hess(progData, x0)
        res = OptimizationMethods.add_identity_until_pd!(
            H0; 
            λ = λ, 
            β = optData.β, 
            max_iterations = optData.hessian_modification_max_iteration
        )
        step = copy(g0)
        if res[2]
            OptimizationMethods.lower_triangle_solve!(step, H0')
            OptimizationMethods.upper_triangle_solve!(step, H0) 
        end
        @test res[2]
        @test optData.λ == res[1] / 2
        @test optData.step ≈ step

        # check backtracing was used
        x = copy(x0)
        F(θ) = OptimizationMethods.obj(progData, θ)
        success = OptimizationMethods.backtracking!(
            x, 
            optData.iter_hist[1],
            F,
            g0,
            step,
            F(x0),
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration)
        @test success
        @test x ≈ x1

        # check the objective history
        @test optData.objective_hist[1] == F(x0)
        @test optData.objective_hist[2] == F(x1)
        @test optData.max_value == F(x0)
        @test optData.max_index == 1
    end # end first iteration test

    # Inductive step: test first maximum update
    max_iterations = window_size
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, 
        hessian_modification_max_iteration = hessian_modification_max_iteration,
        β = β, λ = λ, threshold = 1e-10, 
        max_iterations = max_iterations
    
        # initialize the optimization data for max_iteration - 1
        optData = FixedModifiedNewtonNLSMaxValGD(Float64;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            β = β,
            λ = λ,
            hessian_modification_max_iteration = hessian_modification_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations - 1)

        # get the k-1th iteration of the method and λkm1
        xkm1 = fixed_modified_newton_nls_maxval_gd(optData, progData)
        @test optData.max_value == OptimizationMethods.obj(progData, x0)

        τ_obj_km1 = optData.max_value
        λkm1 = optData.λ

        # initialize the optimization data for max_iteration
        optData = FixedModifiedNewtonNLSMaxValGD(Float64;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            β = β,
            λ = λ,
            hessian_modification_max_iteration = hessian_modification_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations)
        
        # get the kth iteration of the method
        xk = fixed_modified_newton_nls_maxval_gd(optData, progData)

        # check the iterate and gradient histories
        k = optData.stop_iteration

        @test xkm1 == optData.iter_hist[k]
        @test norm(OptimizationMethods.grad(progData, xkm1)) ==
            optData.grad_val_hist[k]

        @test xk == optData.iter_hist[k + 1]
        @test norm(OptimizationMethods.grad(progData, xk)) ==
            optData.grad_val_hist[k + 1]

        # obtain the step taken d_k-1
        gkm1 = OptimizationMethods.grad(progData, xkm1)
        Hkm1 = OptimizationMethods.hess(progData, xkm1)
        res = OptimizationMethods.add_identity_until_pd!(
            Hkm1; 
            λ = λkm1, 
            β = optData.β, 
            max_iterations = optData.hessian_modification_max_iteration
        )
        step = copy(gkm1)
        if res[2]
            OptimizationMethods.lower_triangle_solve!(step, Hkm1')
            OptimizationMethods.upper_triangle_solve!(step, Hkm1) 
        end
        @test res[2]
        @test optData.λ == res[1] / 2
        @test optData.step ≈ step

        # check that backtracking was used
        x = copy(xkm1)
        F(θ) = OptimizationMethods.obj(progData, θ)
        success = OptimizationMethods.backtracking!(
            x,
            xkm1, 
            F,
            gkm1,
            step,
            τ_obj_km1,
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration
        )
        @test success
        @test x ≈ xk

        # check the objective history
        @test optData.objective_hist[1] == F(xk)

        max_val, max_ind = findmax(optData.objective_hist)
        @test optData.max_value == max_val
        @test optData.max_index == max_ind
    end # end arbitrary iteration test

     # Inductive step: arbitrary iteration
     max_iterations = rand((window_size+1):(window_size + 100))
     let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, 
         line_search_max_iteration = line_search_max_iteration, 
         window_size = window_size, 
         hessian_modification_max_iteration = hessian_modification_max_iteration,
         β = β, λ = λ, threshold = 0.0, 
         max_iterations = max_iterations
     
         # initialize the optimization data for max_iteration - 1
         optData = FixedModifiedNewtonNLSMaxValGD(Float64;
             x0 = x0,
             α = α,
             δ = δ,
             ρ = ρ,
             line_search_max_iteration = line_search_max_iteration,
             window_size = window_size,
             β = β,
             λ = λ,
             hessian_modification_max_iteration = hessian_modification_max_iteration,
             threshold = threshold,
             max_iterations = max_iterations - 1)
 
         # get the k-1th iteration of the method and λkm1
         xkm1 = fixed_modified_newton_nls_maxval_gd(optData, progData) 
         τ_obj_km1 = optData.max_value
         λkm1 = optData.λ
 
         # initialize the optimization data for max_iteration
         optData = FixedModifiedNewtonNLSMaxValGD(Float64;
             x0 = x0,
             α = α,
             δ = δ,
             ρ = ρ,
             line_search_max_iteration = line_search_max_iteration,
             window_size = window_size,
             β = β,
             λ = λ,
             hessian_modification_max_iteration = hessian_modification_max_iteration,
             threshold = threshold,
             max_iterations = max_iterations)
         
         # get the kth iteration of the method
         xk = fixed_modified_newton_nls_maxval_gd(optData, progData)
 
         # check the iterate and gradient histories
         k = optData.stop_iteration
 
         @test xkm1 == optData.iter_hist[k]
         @test norm(OptimizationMethods.grad(progData, xkm1)) ==
             optData.grad_val_hist[k]
 
         @test xk == optData.iter_hist[k + 1]
         @test norm(OptimizationMethods.grad(progData, xk)) ==
             optData.grad_val_hist[k + 1]
 
         # obtain the step taken d_k-1
         gkm1 = OptimizationMethods.grad(progData, xkm1)
         Hkm1 = OptimizationMethods.hess(progData, xkm1)
         res = OptimizationMethods.add_identity_until_pd!(
             Hkm1; 
             λ = λkm1, 
             β = optData.β, 
             max_iterations = optData.hessian_modification_max_iteration
         )
         step = copy(gkm1)
         if res[2]
             OptimizationMethods.lower_triangle_solve!(step, Hkm1')
             OptimizationMethods.upper_triangle_solve!(step, Hkm1) 
         end
         @test res[2]
         @test optData.λ == res[1] / 2
         @test optData.step ≈ step
 
         # check that backtracking was used
         x = copy(xkm1)
         F(θ) = OptimizationMethods.obj(progData, θ)
         success = OptimizationMethods.backtracking!(
             x,
             xkm1, 
             F,
             gkm1,
             step,
             τ_obj_km1,
             optData.α,
             optData.δ,
             optData.ρ;
             max_iteration = optData.line_search_max_iteration
         )
         @test success
         @test x ≈ xk
 
         # check the objective history
         @test optData.objective_hist[k + 1] == F(xk)
 
         max_val, max_ind = findmax(optData.objective_hist)
         @test optData.max_value == max_val
         @test optData.objective_hist[optData.max_index] == optData.max_value 
     end # end arbitrary iteration test

    # ############################################################################
    # # Line search failure on first iteration
    # ############################################################################

    # Check to make sure the correct iterate is being returned
    line_search_max_iteration = 0
    let dim = dim, x0 = x0, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, 
        hessian_modification_max_iteration = hessian_modification_max_iteration,
        β = β, λ = λ, threshold = 1e-10, 
        max_iterations = max_iterations
    
        # initialize the optimization data
        optData = FixedModifiedNewtonNLSMaxValGD(Float64;
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            β = β,
            λ = λ,
            hessian_modification_max_iteration = hessian_modification_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations)

        # get first iteration
        x1 = fixed_modified_newton_nls_maxval_gd(optData, progData)

        # check that correct iterate was returned
        @test optData.stop_iteration == 0
        @test x1 == x0
        
        # test histories
        @test optData.iter_hist[1] == x0
        @test optData.grad_val_hist[1] == norm(OptimizationMethods.grad(progData, x0))
        
        # check the objective histories
        @test optData.max_value == OptimizationMethods.obj(progData, x0)
        @test optData.max_index == 1
        @test optData.objective_hist[1] == OptimizationMethods.obj(progData, x0)
        
    end # end first iteration line search failure test
end # end of non-monotone method implementation test cases

end # end of the test cases