# Date: 12/27/2021
# Author: Christian Varner
# Purpose: Implement test cases for quasi-likelihood objective functions

module TestQLLogisticSin

using Test, OptimizationMethods, Random, LinearAlgebra

@testset "Problem -- Quasi-likelihood Logistic Link Sin Variance" begin

    # set the seed for reproducibility
    Random.seed!(1010)

    ####################################
    # Test Struct: QLLogisticSin{T, S}
    ####################################

    # Check if struct is defined 
    @test isdefined(OptimizationMethods, :QLLogisticSin)
    
    # Check supertype 
    @test supertype(OptimizationMethods.QLLogisticSin) == 
        OptimizationMethods.AbstractDefaultQL

    # Test Default Field Names 
    for name in [:meta, :counters, :design, :response]
        @test name in fieldnames(OptimizationMethods.QLLogisticSin)
    end

    # Test Custom Field Names 
    for name in [:mean, :mean_first_derivative, :mean_second_derivative,
        :variance, :variance_first_derivative, :weighted_residual]
        @test name in fieldnames(OptimizationMethods.QLLogisticSin)
    end

    # Test Simulated Data Constructor
    real_types = [Float16, Float32, Float64]
    nobs_default = 1000
    nvar_default = 50 

    for real_type in real_types 
        # Check Assertion
        @test_throws AssertionError OptimizationMethods.QLLogisticSin(
            real_type, nobs=-1, nvar=nvar_default)
        @test_throws AssertionError OptimizationMethods.QLLogisticSin(
            real_type, nobs=nobs_default, nvar=-1)

        #Generate Random Problem 
        progData = OptimizationMethods.QLLogisticSin(real_type, nobs=nobs_default, 
            nvar=nvar_default)

        # Check design type and dimensions 
        @test typeof(progData.design) == Matrix{real_type}
        @test size(progData.design) == (nobs_default, nvar_default)
        
        # Check response type and dimensions 
        @test typeof(progData.response) == Vector{real_type}
        @test length(progData.response) == nobs_default 

        # Check default initial guess and dimensions 
        @test typeof(progData.meta.x0) == Vector{real_type}
        @test length(progData.meta.x0) == nvar_default         
    end

    # Test User-supplied Data Constructor 
    real_types = [Float16, Float32, Float64]
    nobs_default = 1000
    nobs_error = 900
    nvar_default = 50 
    nvar_error = 45

    for real_type in real_types
        # Check assertion: mismatched number of observations 
        @test_throws AssertionError OptimizationMethods.QLLogisticSin(
            randn(real_type, nobs_default, nvar_default),
            randn(real_type, nobs_error)
        )

        # Check assetion: mismatched number of variables 
        @test_throws AssertionError OptimizationMethods.QLLogisticSin(
            randn(real_type, nobs_default, nvar_default), 
            randn(real_type, nobs_default), 
            randn(real_type, nvar_error)
        )
    end


    # test grad
    

    # test hess

end

end