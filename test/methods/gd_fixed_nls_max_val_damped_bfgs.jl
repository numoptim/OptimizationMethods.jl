# Date: 2025/04/29
# Author: Christian Varner
# Purpose: Test the fixed step size damped bfgs method
# with a non-monotone globalization strategy

module TestFixedDampedBFGSNLSMaxVal

using Test, OptimizationMethods, LinearAlgebra, CircularArrays

@testset "Test FixedDampedBFGSNLSMaxValGD{T}" begin

    ############################################################################
    # Test the definition and field names
    ############################################################################

    # test definition
    @test isdefined(OptimizationMethods, :FixedDampedBFGSNLSMaxValGD)

    # define the unique field names
    unique_fields = [:c, :β, :B, :δB, :r, :s, :y, :α, :δ, :ρ, 
        :line_search_max_iteration, :step, :window_size,
        :objective_hist, :max_value, :max_index]
    let fields = unique_fields
        for field in fields
            @test field in fieldnames(FixedDampedBFGSNLSMaxValGD)
        end
    end # end unqiue field tests

    # define the default field names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist, 
        :grad_val_hist, :stop_iteration]
    let fields = default_fields
        for field in fields
            @test field in fieldnames(FixedDampedBFGSNLSMaxValGD)
        end
    end # end default fields tests

    # test to make sure we did not miss a field
    @test length(unique_fields) + length(default_fields) ==
        length(fieldnames(FixedDampedBFGSNLSMaxValGD))

    ############################################################################
    # Error testing
    ############################################################################
    
    ############################################################################
    # Test field types of constructor
    ############################################################################

    field_types(::Type{T}) where {T} =
        [
            (:name, String),
            (:c, T),
            (:β, T),
            (:B, Matrix{T}),
            (:δB, Matrix{T}),
            (:r, Vector{T}),
            (:s, Vector{T}),
            (:y, Vector{T}),
            (:α, T),
            (:δ, T),
            (:ρ, T),
            (:line_search_max_iteration, Int64),
            (:step, Vector{T}),
            (:window_size, Int64),
            (:objective_hist, CircularVector{T, Vector{T}}),
            (:max_value, T),
            (:max_index, Int64),
            (:threshold, T),
            (:max_iterations, Int64),
            (:iter_hist, Vector{Vector{T}}),
            (:grad_val_hist, Vector{T}),
            (:stop_iteration, Int64),
        ]
    real_types = [Float16, Float32, Float64]

    let field_types = field_types, real_types = real_types
        for type in real_types
            for (field_symbol, field_type) in field_types(type)

                # generate a random structure
                optData = FixedDampedBFGSNLSMaxValGD(type;
                    x0 = randn(type, 50),
                    c = rand(type),
                    β = rand(type),
                    α = rand(type),
                    δ = rand(type),
                    ρ = rand(type),
                    line_search_max_iteration = rand(1:100),
                    window_size = rand(1:100),
                    threshold = rand(type),
                    max_iterations = rand(1:100))

                @test field_type == 
                    typeof(getfield(optData, field_symbol))
            end
        end
    end # end the test cases for type information
    
    ############################################################################
    # Test initial values of the constructor
    ############################################################################
    let real_types = real_types
        for type in real_types
            
            # random starting values
            dim = 50
            x0 = randn(type, dim)
            c = rand(type)
            β = rand(type)
            α = rand(type)
            δ = rand(type)
            ρ = rand(type)
            line_search_max_iteration = rand(1:100)
            window_size = rand(1:100)
            threshold = rand(type)
            max_iterations = rand(1:100)

            # generate a random structure
            optData = FixedDampedBFGSNLSMaxValGD(type;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ = δ,
                ρ = ρ,
                line_search_max_iteration = line_search_max_iteration,
                window_size = window_size,
                threshold = threshold,
                max_iterations = max_iterations)

            # test BFGS parameters  
            @test optData.c == c
            @test optData.β == β
            @test size(optData.B) == (dim, dim)
            @test size(optData.δB) == (dim, dim)
            @test length(optData.r) == dim
            @test length(optData.s) == dim
            @test length(optData.y) == dim

            # test line search parameters
            @test optData.α == α
            @test optData.δ == δ
            @test optData.line_search_max_iteration == line_search_max_iteration
            @test length(optData.step) == dim
            
            # test paramters for non-monotone objective cache
            @test optData.window_size == window_size
            @test length(optData.objective_hist) == window_size 
            @test optData.max_value == 0
            @test optData.max_index == -1

            # default parameters
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1
        end
    end

end # end test cases for the struct

@testset "Test fixed_damped_bfgs_nls_maxval_gd -- Monotone" begin

    # random parameters and window_size == 1
    dim = 50
    x0 = randn(dim)
    c = rand()
    β = rand()
    α = rand()
    δ = rand()
    ρ = rand()
    line_search_max_iteration = rand(50:100)
    window_size = 1
    threshold = rand()

    ############################################################################
    # Base Case: max_iteration = 1
    ############################################################################
    max_iterations = 1
    let x0 = x0, c = c, β = β, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        # get random least squares problem
        progData = OptimizationMethods.PoissonRegression(Float64)

        # generate random structure
        optData = FixedDampedBFGSNLSMaxValGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            threshold = threshold,
            max_iterations = max_iterations)

        # get the first iterate (hopefully)
        x1 = fixed_damped_bfgs_nls_maxval_gd(optData, progData)

        # test stop iteration
        @test optData.stop_iteration == 1

        # test iter_hist history and objective hist
        @test optData.iter_hist[optData.stop_iteration + 1] == x1
        @test optData.iter_hist[optData.stop_iteration] == x0 

        # test that x1 was formed by taking the correct step
        g0 = OptimizationMethods.grad(progData, x0)
        B0 = c * norm(g0) * Matrix{Float64}(I, dim, dim)
        step = B0 \ g0

        ## backtracking
        x = copy(x0)
        F(θ) = OptimizationMethods.obj(progData, θ)
        backtrack_success = OptimizationMethods.backtracking!(
            x,
            x0,
            F,
            g0,
            step,
            F(x0),
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration
        )

        # agreement with storage
        @test backtrack_success
        @test optData.step ≈ step
        @test x1 ≈ x

        # test correctness of s, y
        g1 = OptimizationMethods.grad(progData, x1) 
        s = x1 - x0
        y = g1 - g0
        @test optData.s ≈ s
        @test optData.y ≈ y

        # that that δB was formed correctly
        δB1 = zeros(dim, dim)
        B0_copy = copy(B0)
        update_success = OptimizationMethods.update_bfgs!(B0_copy, 
            optData.r, δB1,
            optData.s, optData.y; damped_update = true)
        @test δB1 ≈ optData.δB

        # test δB
        B1 = B0 + optData.δB
        @test optData.B ≈ B1
        
        # test the gradient history 
        @test norm(g0) ≈ optData.grad_val_hist[1]
        @test norm(g1) ≈ optData.grad_val_hist[2]

        # test objective history
        @test optData.objective_hist == [F(x1)]
        @test optData.max_value == F(x1)
        @test optData.max_index == 1
    end

    ############################################################################
    # Inductive Step: random max_iteration
    ############################################################################
    max_iterations = rand(2:20)
    let x0 = x0, c = c, β = β, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, threshold = 0.0,
        max_iterations = max_iterations
        
        # get random least squares problem
        progData = OptimizationMethods.PoissonRegression(Float64)

        # generate random structure
        optData = FixedDampedBFGSNLSMaxValGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            threshold = threshold,
            max_iterations = max_iterations)

        # get the correct iterate (hopefully)
        xk = fixed_damped_bfgs_nls_maxval_gd(optData, progData)
        
        # test stop iteration
        @test optData.stop_iteration == max_iterations
        k = optData.stop_iteration

        # test iter_hist
        @test optData.iter_hist[k + 1] == xk
        xkm1 = optData.iter_hist[k]

        # test that xk was formed correctly
        Bkm1 = optData.B - optData.δB 
        gkm1 = OptimizationMethods.grad(progData, xkm1)
        stepkm1 = Bkm1 \ gkm1
        
        ## backtracking
        x = copy(xkm1)
        F(θ) = OptimizationMethods.obj(progData, θ)
        backtrack_success = OptimizationMethods.backtracking!(
            x,
            xkm1,
            F,
            gkm1,
            stepkm1,
            F(xkm1),
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration
        )

        # test agreement
        @test backtrack_success
        @test optData.step ≈ stepkm1 atol = 1e-5
        @test xk ≈ x

         # test correctness of s, y
         gk = OptimizationMethods.grad(progData, xk) 
         s = xk - xkm1
         y = gk - gkm1
         @test optData.s ≈ s
         @test norm(optData.y - y) ≈ 0 atol = 1e-5

        # that that δB was formed correctly
        δBk = zeros(dim, dim)
        Bkm1_copy = copy(Bkm1)
        update_success = OptimizationMethods.update_bfgs!(Bkm1_copy, 
            optData.r, δBk,
            optData.s, optData.y; damped_update = true)
        @test δBk ≈ optData.δB

        # test δB
        Bk = Bkm1 + optData.δB
        @test optData.B ≈ Bk

        # test the gradient history
        @test norm(optData.grad_val_hist[k + 1] - norm(gk)) ≈ 0 atol = 1e-5

        # test the objective history
        @test optData.objective_hist == [F(xk)]
        @test optData.max_index == 1
        @test optData.max_value == F(xk)
    
    end

    ############################################################################
    # Line search failure at the first step
    ############################################################################
    max_iterations = 1
    let x0 = x0, c = c, β = β, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = 0, 
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        # get random least squares problem
        progData = OptimizationMethods.PoissonRegression(Float64)

        # generate random structure
        optData = FixedDampedBFGSNLSMaxValGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            threshold = threshold,
            max_iterations = max_iterations)

        # get the first iterate (hopefully)
        x1 = fixed_damped_bfgs_nls_maxval_gd(optData, progData)

        # test stop iteration
        @test optData.stop_iteration == 0
        @test x1 ≈ x0
    end

end # end test cases for the function -- monotone

@testset "Test fixed_damped_bfgs_nls_maxval_gd -- Nonmonotone" begin

    # get random parameter with window_size >= 2
    dim = 50
    x0 = randn(dim)
    c = rand()
    β = rand()
    α = rand()
    δ = rand()
    ρ = rand()
    line_search_max_iteration = rand(50:100)
    window_size = rand(2:10)
    threshold = rand()

    # Base Case: max_iteration = 1
    max_iterations = 1
    let x0 = x0, c = c, β = β, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        # get random least squares problem
        progData = OptimizationMethods.PoissonRegression(Float64)

        # generate random structure
        optData = FixedDampedBFGSNLSMaxValGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            threshold = threshold,
            max_iterations = max_iterations)

        # get the first iterate (hopefully)
        x1 = fixed_damped_bfgs_nls_maxval_gd(optData, progData)

        # test stop iteration
        @test optData.stop_iteration == 1

        # test iter_hist history and objective hist
        @test optData.iter_hist[optData.stop_iteration + 1] == x1
        @test optData.iter_hist[optData.stop_iteration] == x0 

        # test that x1 was formed by taking the correct step
        g0 = OptimizationMethods.grad(progData, x0)
        B0 = c * norm(g0) * Matrix{Float64}(I, dim, dim)
        step = B0 \ g0

        ## backtracking
        x = copy(x0)
        F(θ) = OptimizationMethods.obj(progData, θ)
        backtrack_success = OptimizationMethods.backtracking!(
            x,
            x0,
            F,
            g0,
            step,
            F(x0),
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration
        )

        # agreement with storage
        @test backtrack_success
        @test optData.step ≈ step
        @test x1 ≈ x

        # test correctness of s, y
        g1 = OptimizationMethods.grad(progData, x1) 
        s = x1 - x0
        y = g1 - g0
        @test optData.s ≈ s
        @test optData.y ≈ y

        # that that δB was formed correctly
        δB1 = zeros(dim, dim)
        B0_copy = copy(B0)
        update_success = OptimizationMethods.update_bfgs!(B0_copy, 
            optData.r, δB1,
            optData.s, optData.y; damped_update = true)
        @test δB1 ≈ optData.δB

        # test δB
        B1 = B0 + optData.δB
        @test optData.B ≈ B1
        
        # test the gradient history 
        @test norm(g0) ≈ optData.grad_val_hist[1]
        @test norm(g1) ≈ optData.grad_val_hist[2]

        # test objective history
        @test optData.objective_hist[1] == F(x0)
        @test optData.objective_hist[2] == F(x1)
        @test optData.max_value == F(x0)
        @test optData.max_index == 1
    end

    # Inductive Step: max_iteration = window_size
    max_iterations = window_size
    let x0 = x0, c = c, β = β, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, threshold = 0.0,
        max_iterations = max_iterations
        
        # get random least squares problem
        progData = OptimizationMethods.PoissonRegression(Float64)

        # generate random structure
        optData = FixedDampedBFGSNLSMaxValGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            threshold = threshold,
            max_iterations = max_iterations)

        # get the correct iterate (hopefully)
        xk = fixed_damped_bfgs_nls_maxval_gd(optData, progData)
        
        # test stop iteration
        @test optData.stop_iteration == max_iterations
        k = optData.stop_iteration

        # test iter_hist
        @test optData.iter_hist[k + 1] == xk
        xkm1 = optData.iter_hist[k]

        # test that xk was formed correctly
        Bkm1 = optData.B - optData.δB 
        gkm1 = OptimizationMethods.grad(progData, xkm1)
        stepkm1 = Bkm1 \ gkm1
        
        ## backtracking
        x = copy(xkm1)
        F(θ) = OptimizationMethods.obj(progData, θ)
        backtrack_success = OptimizationMethods.backtracking!(
            x,
            xkm1,
            F,
            gkm1,
            stepkm1,
            F(xkm1),
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration
        )

        # test agreement
        @test backtrack_success
        @test optData.step ≈ stepkm1 atol = 1e-5
        @test xk ≈ x

         # test correctness of s, y
         gk = OptimizationMethods.grad(progData, xk) 
         s = xk - xkm1
         y = gk - gkm1
         @test optData.s ≈ s
         @test optData.y ≈ y

        # that that δB was formed correctly
        δBk = zeros(dim, dim)
        Bkm1_copy = copy(Bkm1)
        update_success = OptimizationMethods.update_bfgs!(Bkm1_copy, 
            optData.r, δBk,
            optData.s, optData.y; damped_update = true)
        @test δBk ≈ optData.δB

        # test δB
        Bk = Bkm1 + optData.δB
        @test optData.B ≈ Bk

        # test the gradient history
        @test optData.grad_val_hist[k + 1] ≈ norm(gk)

        # test the objective history
        @test optData.objective_hist[1] == F(xk)

        max_val, max_ind = findmax(optData.objective_hist)
        @test optData.objective_hist[optData.max_index] == max_val
        @test optData.max_value == max_val
    end

    # Inductive Step: random max_iteration
    max_iterations = rand((window_size+1):(window_size + 100))
    let x0 = x0, c = c, β = β, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = line_search_max_iteration, 
        window_size = window_size, threshold = 0.0,
        max_iterations = max_iterations
        
        # get random least squares problem
        progData = OptimizationMethods.PoissonRegression(Float64)

        # generate random structure
        optData = FixedDampedBFGSNLSMaxValGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            threshold = threshold,
            max_iterations = max_iterations - 1)
        
        xkm1 = fixed_damped_bfgs_nls_maxval_gd(optData, progData)
        τkm1 = optData.max_value

        # generate random structure
        optData = FixedDampedBFGSNLSMaxValGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            threshold = threshold,
            max_iterations = max_iterations)

        # get the correct iterate (hopefully)
        xk = fixed_damped_bfgs_nls_maxval_gd(optData, progData)
        
        # test stop iteration
        @test optData.stop_iteration == max_iterations
        k = optData.stop_iteration

        # test iter_hist
        @test optData.iter_hist[k + 1] == xk
        xkm1 = optData.iter_hist[k]

        # test that xk was formed correctly
        Bkm1 = optData.B - optData.δB 
        gkm1 = OptimizationMethods.grad(progData, xkm1)
        stepkm1 = Bkm1 \ gkm1
        
        ## backtracking
        x = copy(xkm1)
        F(θ) = OptimizationMethods.obj(progData, θ)
        backtrack_success = OptimizationMethods.backtracking!(
            x,
            xkm1,
            F,
            gkm1,
            stepkm1,
            τkm1,
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration
        )

        # test agreement
        @test backtrack_success
        @test optData.step ≈ stepkm1 atol = 1e-5
        @test xk ≈ x

         # test correctness of s, y
         gk = OptimizationMethods.grad(progData, xk) 
         s = xk - xkm1
         y = gk - gkm1
         @test norm(optData.s - s) ≈ 0 atol = 1e-10
         @test norm(optData.y - y) ≈ 0 atol = 1e-10

        # that that δB was formed correctly
        δBk = zeros(dim, dim)
        Bkm1_copy = copy(Bkm1)
        update_success = OptimizationMethods.update_bfgs!(Bkm1_copy, 
            optData.r, δBk,
            optData.s, optData.y; damped_update = true)
        @test δBk ≈ optData.δB

        # test δB
        Bk = Bkm1 + optData.δB
        @test optData.B ≈ Bk

        # test the gradient history
        @test optData.grad_val_hist[k + 1] ≈ norm(gk)

        # test the objective history
        @test optData.objective_hist[k + 1] == F(xk)

        max_val, max_ind = findmax(optData.objective_hist)
        @test optData.objective_hist[optData.max_index] == max_val
        @test optData.max_value == max_val
    end

    ############################################################################
    # Line search failure at the first step
    ############################################################################
    max_iterations = 1
    let x0 = x0, c = c, β = β, α = α, δ = δ, ρ = ρ, 
        line_search_max_iteration = 0, 
        window_size = window_size, threshold = threshold,
        max_iterations = max_iterations

        # get random least squares problem
        progData = OptimizationMethods.PoissonRegression(Float64)

        # generate random structure
        optData = FixedDampedBFGSNLSMaxValGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            window_size = window_size,
            threshold = threshold,
            max_iterations = max_iterations)

        # get the first iterate (hopefully)
        x1 = fixed_damped_bfgs_nls_maxval_gd(optData, progData)

        # test stop iteration
        @test optData.stop_iteration == 0
        @test x1 ≈ x0
    end

end # end test cases for the function -- nonmonotone

end # end module