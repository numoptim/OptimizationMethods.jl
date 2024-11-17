# OptimizationMethods.jl 

module TestPoissonRegression 

using Test, ForwardDiff, OptimizationMethods, Random, NLPModels

@testset "Problem: Poission Regression" begin

    # set the seed for reproducibility
    Random.seed!(1010)

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

    
    # Test Generative Constructor
    real_types = [Float16, Float32, Float64]
    nobs_default = 1000
    nvar_default = 50

    for real_type in real_types
        #Check Assertions 
        @test_throws AssertionError OptimizationMethods.PoissonRegression(
            real_type, nobs=-1, nvar=50)
        @test_throws AssertionError OptimizationMethods.PoissonRegression(
            real_type, nobs=1000, nvar=-1)

        #Generate Random Problem
        progData = OptimizationMethods.PoissonRegression(real_type)
        
        #Check design type and dimensions
        @test typeof(progData.design) == Matrix{real_type}
        @test size(progData.design) == (nobs_default, nvar_default)
        
        #Check response type and dimensions 
        @test typeof(progData.response) == Vector{real_type}
        @test length(progData.response) == nobs_default
        
        #Check default initial guess and dimensions
        @test typeof(progData.meta.x0) == Vector{real_type}
        @test length(progData.meta.x0) == nvar_default
    end

    # Test Supplied data Constructor
    real_types = [Float16, Float32, Float64]
    nobs_default = 1000
    nobs_error = 900
    nvar_default = 50
    nvar_error = 45

    for real_type in real_types
        # Check assertion: mismatched number of observations 
        @test_throws AssertionError OptimizationMethods.PoissonRegression(
            randn(real_type, nobs_default, nvar_default),
            randn(real_type, nobs_error)
        )

        # Check assertion: mismatched number of variables 
        @test_throws AssertionError OptimizationMethods.PoissonRegression(
            randn(real_type, nobs_default, nvar_default),
            randn(real_type, nobs_default), 
            x0 = randn(real_type, nvar_error)
        )

        # Mismatched Types Error 
        @test_throws MethodError OptimizationMethods.PoissonRegression(
            randn(real_type, nobs_default, nvar_default), 
            round.(Int64, 20*rand(real_type, nobs_default))
        )
    end

    ####################################
    # Test Struct: Precompute Poisson Regression
    ####################################

    # Check if struct is defined
    @test isdefined(OptimizationMethods, :PrecomputePoissReg)

    # Test super type 
    @test supertype(OptimizationMethods.PrecomputePoissReg) == 
        OptimizationMethods.AbstractPrecompute

    # Test Fields 
    @test :obs_obs_t in fieldnames(OptimizationMethods.PrecomputePoissReg)

    # Test Constructor 
    real_types = [Float16, Float32, Float64]
    nobs_default = 1000
    nvar_default = 50

    for real_type in real_types
        #Generate Random Problem 
        progData = OptimizationMethods.PoissonRegression(real_type)

        #Generate Precompute
        precomp = OptimizationMethods.PrecomputePoissReg(progData)

        #Check Field Type and Dimensions 
        @test typeof(precomp.obs_obs_t) == Array{real_type, 3}
        @test size(precomp.obs_obs_t) == (nobs_default, nvar_default, nvar_default)
        #Compare values 
        @test reduce(+, precomp.obs_obs_t, dims=1)[1,:,:] ≈ 
            progData.design' * progData.design atol=
            eps(real_type) * nobs_default * nvar_default

    end

    ####################################
    # Test Struct: Allocation
    ####################################

    # Check if struct is defined 
    @test isdefined(OptimizationMethods, :AllocatePoissReg)

    # Test super type 
    @test supertype(OptimizationMethods.AllocatePoissReg) == 
        OptimizationMethods.AbstractProblemAllocate

    # Test Fields 
    for name in [:linear_effect, :predicted_rates, :residuals, :grad, :hess]
        @test name in fieldnames(OptimizationMethods.AllocatePoissReg)
    end

    # Test Constructors 
    real_types = [Float16, Float32, Float64]
    nobs_default = 1000 
    nvar_default = 50 

    for real_type in real_types 
        #Generate Random Problem 
        progData = OptimizationMethods.PoissonRegression(real_type)

        #Generate store 
        store = OptimizationMethods.AllocatePoissReg(progData)

        #Check Field Type and Dimension: linear_effect
        @test typeof(store.linear_effect) == Vector{real_type}
        @test length(store.linear_effect) == nobs_default

        #Check Field Type and Dimension: predicted_rates
        @test typeof(store.predicted_rates) == Vector{real_type}
        @test length(store.predicted_rates) == nobs_default

        #Check Field Type and Dimension: residuals
        @test typeof(store.residuals) == Vector{real_type}
        @test length(store.residuals) == nobs_default

        #Check Field Type and Dimension: grad
        @test typeof(store.grad) == Vector{real_type}
        @test length(store.grad) == nvar_default

        #Check Field Type and Dimension: hess
        @test typeof(store.hess) == Matrix{real_type}
        @test size(store.hess) == (nvar_default, nvar_default)
    end

    ####################################
    # Test Method: Initialize
    ####################################
    real_types = [Float16, Float32, Float64]
    nobs_default = 1000 
    nvar_default = 50 

    for real_type in real_types 
        progData = OptimizationMethods.PoissonRegression(real_type)
        precomp, store = OptimizationMethods.initialize(progData)

        @test typeof(precomp) == OptimizationMethods.PrecomputePoissReg{real_type}
        @test typeof(store) == OptimizationMethods.AllocatePoissReg{real_type}
    end

    ####################################
    # Test Methods 
    ####################################
    real_types = [Float16, Float32, Float64]
    nobs_default = 1000 
    nvar_default = 50 
    nargs = 10

    for real_type in real_types

        progData = OptimizationMethods.PoissonRegression(real_type)
        precomp, store =  OptimizationMethods.initialize(progData)
        arg_tests = [randn(real_type, nvar_default) / real_type(sqrt(nvar_default)) 
            for i =1:nargs]

        nevals_obj = 1
        nevals_grad = 1 
        nevals_hess = 1


        function objective(x)
            linear_effect = progData.design * x
            predicted_rates = exp.(linear_effect)
            return sum(predicted_rates) - progData.response'*linear_effect 
        end
        
        ####################################
        # Test Methods: Objective Evaluation 
        ####################################
        for x in arg_tests
            obj = objective(x)

            @test obj ≈ OptimizationMethods.obj(progData, x)
            @test progData.counters.neval_obj == nevals_obj
            nevals_obj += 1

            @test obj ≈ OptimizationMethods.obj(progData, precomp, x)
            @test progData.counters.neval_obj == nevals_obj
            nevals_obj += 1

            @test obj ≈ OptimizationMethods.obj(progData, precomp, store, x)
            @test progData.counters.neval_obj == nevals_obj
            nevals_obj += 1
        end

        ####################################
        # Test Methods: Gradient Evaluation 
        ####################################
        for x in arg_tests
            g = ForwardDiff.gradient(objective, x)

            #TODO: Produces Error 
            #$@test g ≈ OptimizationMethods.grad(progData, x)
            #@test progData.counters.neval_grad == nevals_grad 
            #nevals_grad += 1

            #TODO: Produces Error
            #@test g ≈ OptimizationMethods.grad(progData, precomp, x)
            #@test progData.counters.neval_grad == nevals_grad 
            #nevals_grad += 1

            OptimizationMethods.grad!(progData, precomp, store, x)
            #TODO: Produces Error 
            #@test g ≈ store.grad
            @test progData.counters.neval_grad == nevals_grad 
            nevals_grad += 1
        end

        ####################################
        # Test Methods: Objective-Gradient Evaluation 
        ####################################
        for x in arg_tests
            obj = objective(x)
            gra = ForwardDiff.gradient(objective, x)
            
            # Without Precomputation 
            o, g = OptimizationMethods.objgrad(progData, x)
            @test o ≈ obj
            #TODO: Produces Error 
            #@test g ≈ gra
            @test progData.counters.neval_obj == nevals_obj 
            @test progData.counters.neval_grad == nevals_grad 
            nevals_obj += 1
            nevals_grad += 1


            # With Precomputation 
            o, g = OptimizationMethods.objgrad(progData, precomp, x)
            @test o ≈ obj 
            #TODO: Produces error 
            #@test g ≈ gra
            @test progData.counters.neval_obj == nevals_obj 
            @test progData.counters.neval_grad == nevals_grad 
            nevals_obj += 1
            nevals_grad += 1


            # With Precomputation and Allocation 
            o = OptimizationMethods.objgrad!(progData, precomp, store, x)
            @test o ≈ obj
            #TODO: Produces Error 
            #@test store.grad ≈ gra
            @test progData.counters.neval_obj == nevals_obj 
            @test progData.counters.neval_grad == nevals_grad 
            nevals_obj += 1
            nevals_grad += 1

        end

        ####################################
        # Test Methods: Hessian 
        ####################################
        for x in arg_tests
            h = ForwardDiff.hessian(objective, x)
            
            # Without Precomputation 
            @test h ≈ OptimizationMethods.hess(progData, x)
            @test progData.counters.neval_hess == nevals_hess 
            nevals_hess += 1

            # With Precomputation 
            @test h ≈ OptimizationMethods.hess(progData, precomp, x)
            @test progData.counters.neval_hess == nevals_hess 
            nevals_hess += 1

            # With Precomputation and Allocation 
            #TODO: Produces Errors 
            #OptimizationMethods.hess!(progData, precomp, store, x)
            #@test h ≈ store.hess
            #@test progData.counters.neval_hess == nevals_hess 
            #nevals_hess += 1
        end
    end
end

end