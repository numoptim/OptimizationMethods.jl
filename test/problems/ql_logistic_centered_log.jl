# Date: 02/15/2025
# Author: Christian Varner
# Purpose: Test QLLogisticCenteredLog

module TestQLLogisticCenteredLog

using Test, OptimizationMethods, Random

@testset "Problem Struct -- QLLogisticCenteredLog" begin

    # set seed for reproducibility
    Random.seed!(1010)

    ####################################
    # Test Struct: QLLogisticCenteredLog{T, S}
    ####################################

    # Check if struct is defined 
    @test isdefined(OptimizationMethods, :QLLogisticCenteredLog)
    
    # Check supertype 
    @test supertype(OptimizationMethods.QLLogisticCenteredLog) == 
        OptimizationMethods.AbstractDefaultQL

    # Test Default Field Names 
    for name in [:meta, :counters, :design, :response]
        @test name in fieldnames(OptimizationMethods.QLLogisticCenteredLog)
    end

    # Test Custom Field Names 
    for name in [:mean, :mean_first_derivative, :mean_second_derivative,
        :variance, :variance_first_derivative, :weighted_residual, :p, :c, :d]
        @test name in fieldnames(OptimizationMethods.QLLogisticCenteredLog)
    end

    # Test Simulated Data Constructor
    let real_types = [Float16, Float32, Float64], nobs_default = 1000,
        nvar_default = 50, p_default = 2, c_default = 2, d_default = 2

        for real_type in real_types 
            # Check Assertion
            @test_throws AssertionError OptimizationMethods.QLLogisticCenteredLog(
                real_type, nobs=-1, nvar=nvar_default,
                p = real_type(p_default),
                c = real_type(c_default),
                d = real_type(d_default))
            @test_throws AssertionError OptimizationMethods.QLLogisticCenteredLog(
                real_type, nobs=nobs_default, nvar=-1,
                p = real_type(p_default),
                c = real_type(c_default),
                d = real_type(d_default))

            #Generate Random Problem 
            progData = OptimizationMethods.QLLogisticCenteredLog(real_type, nobs=nobs_default, 
                nvar=nvar_default, p = real_type(p_default), 
                c = real_type(c_default), d = real_type(d_default))

            # Check design type and dimensions 
            @test typeof(progData.design) == Matrix{real_type}
            @test size(progData.design) == (nobs_default, nvar_default)
            
            # Check response type and dimensions 
            @test typeof(progData.response) == Vector{real_type}
            @test length(progData.response) == nobs_default 

            # Check default initial guess and dimensions 
            @test typeof(progData.meta.x0) == Vector{real_type}
            @test length(progData.meta.x0) == nvar_default     
            
            # Check p
            @test typeof(progData.p) == real_type
            @test progData.p == p_default

            # Check c
            @test typeof(progData.c) == real_type
            @test progData.c == c_default

            # Check d
            @test typeof(progData.d) == real_type
            @test progData.d == d_default
        end
    end

    # Test User-supplied Data Constructor
    let real_types = [Float16, Float32, Float64], nobs_default = 1000, 
        nobs_error = 900, nvar_default = 50, nvar_error = 45

        for real_type in real_types
            # Check assertion: mismatched number of observations 
            @test_throws AssertionError OptimizationMethods.QLLogisticCenteredLog(
                randn(real_type, nobs_default, nvar_default),
                randn(real_type, nobs_error)
            )

            # Check assetion: mismatched number of variables 
            @test_throws AssertionError OptimizationMethods.QLLogisticCenteredLog(
                randn(real_type, nobs_default, nvar_default), 
                randn(real_type, nobs_default);
                x0 = randn(real_type, nvar_error)
            )
        end
    end
end

end