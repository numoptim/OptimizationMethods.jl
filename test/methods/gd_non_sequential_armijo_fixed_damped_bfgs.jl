# Date: 2025/04/30
# Author: Christian Varner
# Purpose: Test cases for the damped BFGS quasi-newton method
# with fixed step sizes globalized through the non-sequential
# Armijo framework

module TestNonsequentialArmijoDampedBFGS

using Test, OptimizationMethods, CircularArrays, LinearAlgebra, Random

@testset "Test NonsequentialArmijoFixedDampedBFGSGD" begin

    ############################################################################
    # Test definition and field names
    ############################################################################
    
    # definition
    @test isdefined(OptimizationMethods, :NonsequentialArmijoFixedDampedBFGSGD)

    # unique names
    unique_field = [
        :∇F_θk,
        :B_θk,
        :Bjk,
        :δBjk,
        :rjk,
        :sjk,
        :yjk,
        :djk,
        :c,
        :β,
        :norm_∇F_ψ,
        :α,
        :δk,
        :δ_upper,
        :ρ,
        :objective_hist,
        :reference_value,
        :reference_value_index,
        :acceptance_cnt,
        :τ_lower,
        :τ_upper,
        :inner_loop_radius,
        :inner_loop_max_iterations]

    let fields = unique_field
        for field in fields
            @test field in fieldnames(NonsequentialArmijoFixedDampedBFGSGD)
        end
    end # end test cases for unique fields

    # default fields
    default_fields = [:name, :threshold, :max_iterations, :iter_hist, 
        :grad_val_hist, :stop_iteration]
    
    let fields = default_fields
        for field in fields
            @test field in fieldnames(NonsequentialArmijoFixedDampedBFGSGD)
        end
    end # end teest cases for default fields

    ############################################################################
    # Test error throwing in constructor
    ############################################################################
    
    ############################################################################
    # Test field types
    ############################################################################

    field_info(::Type{T}) where {T} = 
        [
            (:name, String),
            (:∇F_θk, Vector{T}),
            (:B_θk, Matrix{T}),
            (:Bjk, Matrix{T}),
            (:δBjk, Matrix{T}),
            (:rjk, Vector{T}),
            (:sjk, Vector{T}),
            (:yjk, Vector{T}),
            (:djk, Vector{T}),
            (:c, T),
            (:β, T),
            (:norm_∇F_ψ, T),
            (:α, T),
            (:δk, T),
            (:δ_upper, T),
            (:ρ, T),
            (:objective_hist, CircularVector{T, Vector{T}}),
            (:reference_value, T),
            (:reference_value_index, Int64),
            (:acceptance_cnt, Int64),
            (:τ_lower, T),
            (:τ_upper, T),
            (:inner_loop_radius, T),
            (:inner_loop_max_iterations, Int64),
            (:threshold, T),
            (:max_iterations, Int64),
            (:iter_hist, Vector{Vector{T}}),
            (:grad_val_hist, Vector{T}),
            (:stop_iteration, Int64),
        ]

    let field_info = field_info, real_types = [Float16, Float32, Float64]
        dim = 50
        for type in real_types
            
            # generate a random structure
            x0 =               randn(type, dim)
            c =                 rand(type)
            β =                 rand(type)
            α =                 rand(type)
            δ0 =                rand(type)
            ρ =                 rand(type)
            M =                 rand(1:10)
            threshold =         randn(type)
            inner_loop_radius = rand(type)
            inner_loop_max_iterations = rand(1:100)
            max_iterations = rand(1:100)
            δ_upper = δ0 + 1
            
            # construct
            optData = NonsequentialArmijoFixedDampedBFGSGD(type;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations)

            # test the field types
            for (field_symbol, field_type) in field_info(type)
                @test field_type == typeof(getfield(optData, field_symbol))
            end

        end

    end # end the test cases for the field info

    ############################################################################
    # Test initial values for constructor
    ############################################################################

    let real_types = [Float16, Float32, Float64], dim = 50
        for type in real_types
            
            x0 =               randn(type, dim)
            c =                 rand(type)
            β =                 rand(type)
            α =                 rand(type)
            δ0 =                rand(type)
            ρ =                 rand(type)
            M =                 rand(1:10)
            threshold =         randn(type)
            inner_loop_radius = rand(type)
            inner_loop_max_iterations = rand(1:100)
            max_iterations = rand(1:100)
            δ_upper = δ0 + 1
            
            # construct
            optData = NonsequentialArmijoFixedDampedBFGSGD(type;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations)

            @test length(optData.∇F_θk) == dim
            @test length(optData.rjk) == dim
            @test length(optData.sjk) == dim
            @test length(optData.yjk) == dim
            @test length(optData.djk) == dim
            @test size(optData.B_θk) == (dim, dim)
            @test size(optData.δBjk) == (dim, dim)
            
            @test optData.c == c
            @test optData.β == β
            @test optData.α == α
            @test optData.δk == δ0
            @test optData.ρ == ρ
            @test optData.norm_∇F_ψ == type(0)
            
            @test length(optData.objective_hist) == M
            @test optData.reference_value == type(0)
            @test optData.reference_value_index == -1
            @test optData.acceptance_cnt == 0

            @test optData.τ_lower == type(-1) 
            @test optData.τ_upper == type(-1)

            @test optData.inner_loop_radius == inner_loop_radius
            @test optData.inner_loop_max_iterations == inner_loop_max_iterations
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1
        end
    end # end test for initial values

end # end test on structure

@testset "Utility -- Update Algorithm Parameters NonseqArmijoDampedBFGS" begin

    include("../utility/update_algorithm_parameters_test_cases.jl")

    # Random arguments
    dim = 50
    x0 =               randn(dim)
    c =                 rand()
    β =                 rand()
    α =                 rand()
    δ0 =                rand()
    ρ =                 rand()
    M =                 rand(1:10)
    threshold =        randn()
    inner_loop_radius = rand()
    inner_loop_max_iterations = rand(1:100)
    max_iterations = rand(1:100)
    δ_upper = δ0 + 1
 
    # build structure
    optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
        x0 = x0,
        c = c,
        β = β,
        α = α,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        M = M,
        inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold,
        max_iterations = max_iterations)

    # conduct test cases
    update_algorithm_parameters_test_cases(optData, dim, max_iterations)

end # end test on update algorithm parameters

@testset "Utility -- Inner Loop NonseqArmijoDampedBFGS" begin

    # random arguments
    dim = 50
    x0 =               randn(dim)
    c =                 rand()
    β =                 rand()
    α =                 rand()
    δ0 =                rand()
    ρ =                 rand()
    M =                 rand(1:10)
    threshold =        randn()
    inner_loop_radius = rand()
    inner_loop_max_iterations = rand(1:100)
    max_iterations = rand(1:100)
    δ_upper = δ0 + 1   
    
    # build structure
    optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
        x0 = x0,
        c = c,
        β = β,
        α = α,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        M = M,
        inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold,
        max_iterations = max_iterations)
    
    progData = OptimizationMethods.LogisticRegression(Float64, nvar=50)
    precomp, store = OptimizationMethods.initialize(progData)
    ks = [1, max_iterations]

    for k in ks
        # Test first event trigger: radius violation
        let ψjk=x0 .+ inner_loop_radius, 
            θk=x0, optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 1.5
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k;
                radius = optData.inner_loop_radius,
                max_iteration=100)

            @test ψjk == x0 .+ inner_loop_radius
        end # end first event trigger test

         # Test second event trigger: τ_lower 
         let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 0.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k;
                radius = optData.inner_loop_radius,
                max_iteration=100)

            @test ψjk == x0
        end # end second event triggering event

        # Test third event trigger: τ_upper 
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 2.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k;
                radius = optData.inner_loop_radius,
                max_iteration=100)

            @test ψjk == x0
        end

         # Test fourth event trigger: max_iteration
         let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 1.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k; 
                radius = optData.inner_loop_radius,
                max_iteration=0)

            @test ψjk == x0
        end

        # Test first iteration
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            j=1

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)

            optData.δk = 1.5
            α = optData.α

            # compute the initial step
            g00 = OptimizationMethods.grad(progData, θk)
            B00 = optData.c * norm(g00) * Matrix{Float64}(I, 50, 50)
            step = optData.δk * optData.α * (B00 \ g00)

            # reset
            optData.Bjk = B00
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k; radius = optData.inner_loop_radius, max_iteration = 1
            )
            @test j == 1

            # test cases
            @test ψjk ≈ θk - step 
            @test optData.α == α
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ == norm(store.grad)

            g10 = OptimizationMethods.grad(progData, ψjk)
            @test optData.sjk ≈ ψjk - θk
            @test optData.yjk ≈ g10 - g00

            # test to make sure damped BFGS was updated
            update_success = OptimizationMethods.update_bfgs!(
                B00, optData.rjk, optData.δBjk,
                optData.sjk, optData.yjk; damped_update = true)
            OptimizationMethods.add_identity(optData.Bjk, optData.β)

            @test B00 ≈ optData.Bjk
        end

         # test a random iteration
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            #To do this test correctly, we would need to know at what iteration 
            #j an inner loop exists.
            max_iteration = rand(2:100)

            # Reset
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.Bjk = optData.c * norm(store.grad) * Matrix{Float64}(I, 50, 50)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1.5

            #Get exit iteration j -- radius == 10 so that we get resonable j
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k; max_iteration = max_iteration
            )

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.Bjk = optData.c * norm(store.grad) * Matrix{Float64}(I, 50, 50)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1.5

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = j-1
            )
            α = optData.α

            ψ_jm1_k = copy(ψjk)
            B_jm1_k = copy(optData.Bjk)
            step = optData.δk * optData.α * (optData.Bjk \ store.grad)

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.Bjk = optData.c * norm(store.grad) * Matrix{Float64}(I, 50, 50)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1.5

            #Get ψ_{j,k}
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = max_iteration
            )

            @test ψjk ≈ ψ_jm1_k - step
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ ≈ norm(store.grad)
            @test optData.α == α

            # test the values of s, y
            gjm10 = OptimizationMethods.grad(progData, ψ_jm1_k) 
            gj0 = OptimizationMethods.grad(progData, ψjk)
            @test optData.sjk ≈ ψjk - ψ_jm1_k
            @test optData.yjk ≈ gj0 - gjm10

           # test to make sure damped BFGS was updated
           update_success = OptimizationMethods.update_bfgs!(
                B_jm1_k, optData.rjk, optData.δBjk,
                optData.sjk, optData.yjk; damped_update = true)
            OptimizationMethods.add_identity(B_jm1_k, optData.β)

            @test B_jm1_k ≈ optData.Bjk 

        end # end of the test of a random iteration
    end # end the test for the initial iteration

end # end test on inner loop

@testset "Test nonsequential_armijo_fixed_damped_bfgs Monotone" begin
end # end test for monotone 

@testset "Test nonsequential_armijo_fixed_damped_bfgs Nonmonotone" begin
end # end test for nonmonotone

end # End module