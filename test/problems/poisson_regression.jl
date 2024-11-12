# OptimizationMethods.jl 

module TestPoissonRegression 

using Test, ForwardDiff, OptimizationMethods, Random

@testset "Problem: Poission Regression" begin

    # set the seed for reproducibility
    Random.seed(1010)

    ####################################
    # Test Struct: Poisson Regression
    ####################################

    # Check if struct is defined 
    @test isdefined(OptimizationMethods, :PoissonRegression)

    # Test Super type 
    @test supertype(OptimizationMethods.PoissonRegression) == AbstractNLPModel

    # Test Fields 
    for name in [:meta, :counters, :design, :response]
        @test name in fieldnames(OptimizationMethods.PoissonRegression)
    end

    ####################################
    # Test constructors
    ####################################

    real_types = [Float16, Float32, Float64]

    for real_type in real_types
        #Check Assertions 
        @test_throws AssertionError OptimizationMethods.PoissonRegression(
            real_type, nobs=-1, nvar=50)
        @test_throws AssertionError OptimizationMethods.PoissonRegression(
            real_type, nobs=1000, nvar=-1)

        #Generate Random Problem
        progData = OptimizationMethods.PoissonRegression(real_type)
        
        #TODO: What are sensible tests for a constructor
    end

end

end