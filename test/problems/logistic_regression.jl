# Date: 10/05/2024
# Author: Christian Varner
# Purpose: Implement some test cases for logistic regression

module TestPoissonRegression

using Test, OptimizationMethods, Random, LinearAlgebra, NLPModels

@testset "Problem: Logistic Regression" begin

    # set the seed for reproducibility
    Random.seed!(1010)

    ####################################
    # Test Struct: Logistic Regression
    ####################################

    # check if struct is defined
    @test isdefined(OptimizationMethods, :LogisticRegression)

    # test super type
    @test supertype(OptimizationMethods.LogisticRegression) == AbstractNLPModel

    # test fields
    field_names = [:meta, :counters, :design, :response]

    for name in field_names
        @test name in fieldnames(OptimizationMethods.LogisticRegression)
    end

    # test field values?

    ####################################
    # Test constructors
    ####################################

    # Test constructor 1 -- default values
    real_types = [Float16, Float32, Float64] # TODO: need to figure out why ints aren't working
    nobs = 1000
    nvar = 50

    for default in [true, false]
        nobs = default ? 1000 : 125
        nvar = default ? 50 : 10
        for real_type in real_types
            
            progData = nothing
            if default
                progData = OptimizationMethods.LogisticRegression(real_type)
            else
                progData = OptimizationMethods.LogisticRegression(
                    real_type, 
                    nobs = 125, 
                    nvar = 10
                )
            end

            # test types
            @test typeof(progData.design) == Matrix{real_type}
            @test typeof(progData.response) == Vector{Bool}
            @test typeof(progData.meta.x0) == Vector{real_type}
            
            # test values
            @test progData.meta.nvar == nvar
            @test progData.meta.name == "Logistic Regression"
            @test progData.meta.x0 == ones(real_type, nvar) ./ real_type(sqrt(nvar))

            @test size(progData.design) == (nobs, nvar)
            @test size(progData.response, 1) == nobs
        end
    end

    # test constructor 2
    for default_x0 in [true, false]
        for real_type in real_types
            nobs = 100
            nvar = 10 
            A = randn(real_type, nobs, nvar)
            b = Vector{Bool}(bitrand(nobs))
            x0 = ones(real_type, nvar) ./ real_type(sqrt(nvar))
            
            progData = nothing
            if default_x0
                progData = OptimizationMethods.LogisticRegression(
                    A, b    
                )
            else
                x0 = randn(real_type, nvar)
                progData = OptimizationMethods.LogisticRegression(
                    A, b; x0 = x0
                )
            end
            # test types
            @test typeof(progData.design) == Matrix{real_type}
            @test typeof(progData.response) == Vector{Bool}
            @test typeof(progData.meta.x0) == Vector{real_type}
            
            # test values
            @test progData.meta.nvar == nvar
            @test progData.meta.name == "Logistic Regression"

            # test values
            @test progData.meta.x0 == x0 
            @test progData.design == A
            @test progData.response == b
        
        end
    end


    ####################################
    # Test Struct Precomputed
    ####################################

    # TODO

    ####################################
    # Test Allocated struct
    ####################################
    # TODO

    ####################################
    # Test initialize 
    ####################################
    # TODO

    ####################################
    # Test functionality - group 1
    ####################################

    # TODO

    ####################################
    # Test functionality - group 2
    ####################################

    # TODO
    
    ####################################
    # Test functionality - group 3
    ####################################

    # TODO
end

end # end module