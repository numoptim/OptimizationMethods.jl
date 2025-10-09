# Date: 2025/10/09
# Author: Christian Varner
# Purpose: Test QLLogistic.jl

module TestQLLogistic

using Test, OptimizationMethods, Random

@testset "Problem Struct -- QLLogistic" begin

    # set seed for reproducibility
    Random.seed!(1010)

    ####################################
    # Test Struct: QLLogistic{T, S}
    ####################################

    # Check if struct is defined 
    @test isdefined(OptimizationMethods, :QLLogistic)
    
    # Check supertype 
    @test supertype(OptimizationMethods.QLLogistic) == 
        OptimizationMethods.AbstractDefaultQL

    # Test Default Field Names 
    for name in [:meta, :counters, :design, :response]
        @test name in fieldnames(OptimizationMethods.QLLogistic)
    end

    # Test Custom Field Names 
    for name in [:β_true, :mean, :mean_first_derivative, :mean_second_derivative,
        :variance, :variance_first_derivative, :weighted_residual]
        @test name in fieldnames(OptimizationMethods.QLLogistic)
    end

    # Test Simulated Data Constructor
    let real_types = [Float32, Float64], nobs_default = 1000,
        nvar_default = 50, 
        V = OptimizationMethods.logistic_variance,
        dV = OptimizationMethods.dlogistic_variance
        
        a = .5 * rand()
        for real_type in real_types

                # Initialize QLLogistic Struct
                progData = OptimizationMethods.QLLogistic(real_type,
                V, dV; nobs = nobs_default, nvar = nvar_default, 
                a = a, vmax = 0.25)

                # Check design is correctly initialized
                @test typeof(progData.design) == Matrix{real_type}
                @test size(progData.design) == (nobs_default, nvar_default)
                @test all(progData.design .>= 0.0) && all(progData.design .<= 1.0)

                # Check β_true is correctly initialized
                @test typeof(progData.β_true) == Vector{real_type}
                @test size(progData.β_true) == (nvar_default,)

                lb = OptimizationMethods.inverse_logistic(a)/nvar_default
                ub = OptimizationMethods.inverse_logistic(1-a)/nvar_default
                @test all(progData.β_true .>= lb) && all(progData.β_true .<= ub)

                # Check response is correctly initialized
                @test typeof(progData.response) == Vector{real_type}
                @test size(progData.response) == (nobs_default,)
                @test all(progData.response .>= 0.0) && all(progData.response .<= 1.0)

                # Check mean functions are correctly initialized
                @test progData.mean === OptimizationMethods.logistic
                @test progData.mean_first_derivative === OptimizationMethods.dlogistic
                @test progData.mean_second_derivative === OptimizationMethods.ddlogistic

                # Check variance functions are correctly initialized
                @test progData.variance === V
                @test progData.variance_first_derivative === dV
        end

    end # end let block

     # Test Constructor with User Provided Data
    let real_types = [Float32, Float64], 
        nobs_default = 1000,
        nobs_error = 900,
        nvar_default = 50, 
        nvar_error = 45,
        V = OptimizationMethods.logistic_variance,
        dV = OptimizationMethods.dlogistic_variance

        for real_type in real_types

            # Create error with nobs_error
            design = randn(real_type, nobs_error, nvar_default)
            response = randn(real_type, nobs_default)
            @test_throws AssertionError OptimizationMethods.QLLogistic(
                design, response, V, dV)

            # Create error with nvar_error
            design = randn(real_type, nobs_default, nvar_error)
            x0 = randn(real_type, nvar_default)
            response = randn(real_type, nobs_default)
            @test_throws AssertionError OptimizationMethods.QLLogistic(
                design, response, V, dV; x0 = x0)

            # Generate correct data
            design = randn(real_type, nobs_default, nvar_default)
            response = randn(real_type, nobs_default)
            progData = OptimizationMethods.QLLogistic(
                design, response, V, dV; x0 = x0)

            # Check design is correctly initialized
            @test typeof(progData.design) == Matrix{real_type}
            @test size(progData.design) == (nobs_default, nvar_default)
            @test progData.design == design

            # Check β_true is correctly initialized
            @test progData.β_true === nothing

            # Check response is correctly initialized
            @test typeof(progData.response) == Vector{real_type}
            @test size(progData.response) == (nobs_default,)
            @test progData.response == response

            # Check mean functions are correctly initialized
            @test progData.mean === OptimizationMethods.logistic
            @test progData.mean_first_derivative === OptimizationMethods.dlogistic
            @test progData.mean_second_derivative === OptimizationMethods.ddlogistic

            # Check variance functions are correctly initialized
            @test progData.variance === V
            @test progData.variance_first_derivative === dV
        end
    end # end let block

end # end test set

end # end module