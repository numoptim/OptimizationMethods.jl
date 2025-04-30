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
end # end test on inner loop

@testset "Test nonsequential_armijo_fixed_damped_bfgs Monotone" begin
end # end test for monotone 

@testset "Test nonsequential_armijo_fixed_damped_bfgs Nonmonotone" begin
end # end test for nonmonotone

end # End module