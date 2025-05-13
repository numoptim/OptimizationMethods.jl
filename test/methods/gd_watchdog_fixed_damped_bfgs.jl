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
end

@testset "Test WatchdogFixedDampedBFGSGD{T} Method Monotone" begin
end

@testset "Test WatchdogFixedDampedBFGSGD{T} Method Nonmonotone" begin
end

end