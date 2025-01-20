# Date: 01/20/2025
# Author: Christian Varner
# Purpose: Testing the implementation for the proximal point
# subproblem

module TestProximalPointSubproblem

using Test, ForwardDiff, OptimizationMethods, NLPModels, LinearAlgebra, Random

@testset "Problem: Proximal Point Subproblem" begin

    # set seed
    Random.seed!(1010)

    ###########################################################################
    # Test Struct: Proximal Point Subproblem
    ###########################################################################

    # check if struct is defined
    @test isdefined(OptimizationMethods, :ProximalPointSubproblem)

    # test super type
    @test supertype(OptimizationMethods.ProximalPointSubproblem) == 
        AbstractNLPModel

    # test fields
    fields = [:meta, :counters, :progData, :progData_precomp, 
        :progData_store, :penalty, :θkm1]
    for name in fields
        @test name in fieldnames(OptimizationMethods.ProximalPointSubproblem)
    end
    @test length(fields) == 
        length(fieldnames(OptimizationMethods.ProximalPointSubproblem))

    # test the constructor
    real_types = [Float16, Float32, Float64]
    nobs = 1000
    nvar = 50

    let real_types = real_types, nobs = nobs, nvar = nvar
        for T in real_types

            # initialize problem used to construct proximal point subproblem
            subProgData = OptimizationMethods.LeastSquares(T)
            subPrecomp, subStore = OptimizationMethods.initialize(subProgData)

            # random penalty and initial starting point
            penalty = randn(T, 1)[1]
            θkm1 = randn(T, nvar)

            # construct proximal point subproblem
            progData = OptimizationMethods.ProximalPointSubproblem(T;
                progData = subProgData,
                progData_precomp = subPrecomp,
                progData_store = subStore,
                penalty = penalty,
                θkm1 = θkm1
                )

            # test field types
            @test typeof(progData.meta) == NLPModelMeta{T, Vector{T}}
            @test typeof(progData.counters) == Counters
            @test typeof(progData.progData) == 
                OptimizationMethods.LeastSquares{T, Vector{T}}

            @test typeof(progData.progData_precomp) == 
                OptimizationMethods.PrecomputeLS{T}

            @test typeof(progData.progData_store) == 
                OptimizationMethods.AllocateLS{T}

            @test typeof(penalty) == T
            @test typeof(θkm1) == Vector{T}

            # test field values

            ## check progData.progData
            @test progData.progData.coef == subProgData.coef
            @test progData.progData.cons == subProgData.cons

            ## check progData.progData_precomp
            @test progData.progData_precomp.coef_t_coef ==
                subPrecomp.coef_t_coef
            @test progData.progData_precomp.coef_t_cons ==
                subPrecomp.coef_t_cons
            @test progData.progData_precomp.cons_t_cons ==
                subPrecomp.cons_t_cons

            ## check progData.progData_store
            @test progData.progData_store.res == zeros(T, nobs)
            @test progData.progData_store.jac == 
                subStore.jac

            @test progData.progData_store.grad == zeros(T, nvar)
            @test progData.progData_store.hess ==
                subPrecomp.coef_t_coef 

            ## check penalty and θkm1
            @test progData.penalty == penalty
            @test progData.θkm1 == θkm1
        end
    end

    ###########################################################################
    # Test Struct: Proximal Point Subproblem Precompute
    ###########################################################################

    # test definition
    @test isdefined(OptimizationMethods, :PrecomputeProximalPointSubproblem)

    # test supertype
    @test supertype(OptimizationMethods.PrecomputeProximalPointSubproblem) ==
        OptimizationMethods.AbstractPrecompute

    # test field values
    @test length(fieldnames(
        OptimizationMethods.PrecomputeProximalPointSubproblem)) == 0

    ###########################################################################
    # Test Struct: Proximal Point Subproblem Allocate
    ###########################################################################

    # test definition
    @test isdefined(OptimizationMethods, :AllocateProximalPointSubproblem)

    # test supertype
    @test supertype(OptimizationMethods.AllocateProximalPointSubproblem) ==
        OptimizationMethods.AbstractProblemAllocate

    # test field values
    fields = [:grad, :hess]
    for name in fields
        @test name in fieldnames(OptimizationMethods.AllocateProximalPointSubproblem)
    end

    # test constructor
    real_type = [Float16, Float32, Float64]
    nobs = 1000
    nvar = 50

    let real_type = real_type, nobs = nobs, nvar = nvar
        for T in real_type

            # initialize problem used to construct proximal point subproblem
            subProgData = OptimizationMethods.LeastSquares(T)
            subPrecomp, subStore = OptimizationMethods.initialize(subProgData)

            # random penalty and initial starting point
            penalty = randn(T, 1)[1]
            θkm1 = randn(T, nvar)

            # construct proximal point subproblem
            progData = OptimizationMethods.ProximalPointSubproblem(T;
                progData = subProgData,
                progData_precomp = subPrecomp,
                progData_store = subStore,
                penalty = penalty,
                θkm1 = θkm1
                )

            # Allocate problem constructor
            store = OptimizationMethods.AllocateProximalPointSubproblem(progData)

            # test values
            @test typeof(store.grad) == Vector{T}
            @test typeof(store.hess) == Matrix{T}

            # test values
            @test store.grad == zeros(T, nvar)
            @test store.hess == zeros(T, nvar, nvar)
            
        end
    end

    ###########################################################################
    # Test Utility: initialize()
    ###########################################################################

    real_type = [Float16, Float32, Float64]
    nobs = 1000
    nvar = 50

    let real_type = real_type, nobs = nobs, nvar = nvar
        for T in real_type

            # initialize problem used to construct proximal point subproblem
            subProgData = OptimizationMethods.LeastSquares(T)
            subPrecomp, subStore = OptimizationMethods.initialize(subProgData)

            # random penalty and initial starting point
            penalty = randn(T, 1)[1]
            θkm1 = randn(T, nvar)

            # construct proximal point subproblem
            progData = OptimizationMethods.ProximalPointSubproblem(T;
                progData = subProgData,
                progData_precomp = subPrecomp,
                progData_store = subStore,
                penalty = penalty,
                θkm1 = θkm1
                )

            # initialize problem data
            output = OptimizationMethods.initialize(progData)

            # test length of output and types
            @test length(output) == 2
            @test typeof(output[1]) == 
                OptimizationMethods.PrecomputeProximalPointSubproblem{T}

            @test typeof(output[2]) ==
                OptimizationMethods.AllocateProximalPointSubproblem{T}
            
            precomp, store = output[1], output[2]

            # test values
            @test typeof(store.grad) == Vector{T}
            @test typeof(store.hess) == Matrix{T}

            # test values
            @test store.grad == zeros(T, nvar)
            @test store.hess == zeros(T, nvar, nvar)
            
        end
    end

    ###########################################################################
    # Test Utility: Proximal Point Functionality
    ###########################################################################

    # methods that do not use precomp or store
    real_type = [Float64]
    nobs = 1000
    nvar = 50

    let real_type = real_type, nobs = nobs, nvar = nvar
        for T in real_type

            # initialize problem used to construct proximal point subproblem
            subProgData = OptimizationMethods.LeastSquares(T)
            subPrecomp, subStore = OptimizationMethods.initialize(subProgData)

            # random penalty and initial starting point
            penalty = randn(T, 1)[1]
            θkm1 = randn(T, nvar)

            # construct proximal point subproblem
            progData = OptimizationMethods.ProximalPointSubproblem(T;
                progData = subProgData,
                progData_precomp = subPrecomp,
                progData_store = subStore,
                penalty = penalty,
                θkm1 = θkm1
                )

            # ForwardDiff functions
            A = subProgData.coef
            b = subProgData.cons
            F(θ) = .5 * (norm(A * θ - b)^2) + .5 * penalty * norm(θ - θkm1)^2
            G(θ) = ForwardDiff.gradient(F, θ)
            H(θ) = ForwardDiff.hessian(F, θ)

            ####################################################################
            # TESTS FOR FUNCTIONALITY
            ####################################################################
            test_point = randn(T, nvar)

            ## objective
            @test F(test_point) ≈ OptimizationMethods.obj(progData, test_point)

            ## gradient            
            @test G(test_point) ≈ 
                OptimizationMethods.grad(progData, test_point)

            ## objgrad
            output = OptimizationMethods.objgrad(progData, test_point)
            @test length(output) == 2
            @test output[1] ≈ F(test_point)
            @test output[2] ≈ G(test_point)

            ## hess
            @test H(test_point) ≈ OptimizationMethods.hess(progData, test_point)
        end
    end

    # methods that use precomp but not store
    real_type = [Float64]
    nobs = 1000
    nvar = 50

    let real_type = real_type, nobs = nobs, nvar = nvar
        for T in real_type

            # initialize problem used to construct proximal point subproblem
            subProgData = OptimizationMethods.LeastSquares(T)
            subPrecomp, subStore = OptimizationMethods.initialize(subProgData)

            # random penalty and initial starting point
            penalty = randn(T, 1)[1]
            θkm1 = randn(T, nvar)

            # construct proximal point subproblem
            progData = OptimizationMethods.ProximalPointSubproblem(T;
                progData = subProgData,
                progData_precomp = subPrecomp,
                progData_store = subStore,
                penalty = penalty,
                θkm1 = θkm1
                )

            precomp, store = OptimizationMethods.initialize(progData)

            # ForwardDiff functions
            A = subProgData.coef
            b = subProgData.cons
            F(θ) = .5 * (norm(A * θ - b)^2) + .5 * penalty * norm(θ - θkm1)^2
            G(θ) = ForwardDiff.gradient(F, θ)
            H(θ) = ForwardDiff.hessian(F, θ)

            ####################################################################
            # TESTS FOR FUNCTIONALITY
            ####################################################################
            test_point = randn(T, nvar)

            ## objective
            @test F(test_point) ≈
                OptimizationMethods.obj(progData, precomp, test_point)

            ## gradient            
            @test G(test_point) ≈ 
                OptimizationMethods.grad(progData, precomp, test_point)

            ## objgrad
            output = OptimizationMethods.objgrad(progData, precomp, test_point)
            @test length(output) == 2
            @test output[1] ≈ F(test_point)
            @test output[2] ≈ G(test_point)

            ## hess
            @test H(test_point) ≈ 
                OptimizationMethods.hess(progData, precomp, test_point)
        end
    end

    # methods that use precomp and store
    real_type = [Float64]
    nobs = 1000
    nvar = 50

    let real_type = real_type, nobs = nobs, nvar = nvar
        for T in real_type

            # initialize problem used to construct proximal point subproblem
            subProgData = OptimizationMethods.LeastSquares(T)
            subPrecomp, subStore = OptimizationMethods.initialize(subProgData)

            # random penalty and initial starting point
            penalty = randn(T, 1)[1]
            θkm1 = randn(T, nvar)

            # construct proximal point subproblem
            progData = OptimizationMethods.ProximalPointSubproblem(T;
                progData = subProgData,
                progData_precomp = subPrecomp,
                progData_store = subStore,
                penalty = penalty,
                θkm1 = θkm1
                )

            precomp, store = OptimizationMethods.initialize(progData)

            # ForwardDiff functions
            A = subProgData.coef
            b = subProgData.cons
            F(θ) = .5 * (norm(A * θ - b)^2) + .5 * penalty * norm(θ - θkm1)^2
            G(θ) = ForwardDiff.gradient(F, θ)
            H(θ) = ForwardDiff.hessian(F, θ)

            ####################################################################
            # TESTS FOR FUNCTIONALITY
            ####################################################################
            test_point = randn(T, nvar)

            ## objective
            @test F(test_point) ≈
                OptimizationMethods.obj(progData, precomp, store, test_point)

            ## gradient       
            OptimizationMethods.grad!(progData, precomp, store, test_point) 
            @test norm(subStore.grad - 
                OptimizationMethods.grad(subProgData, test_point)) ≈ 0 atol = 1e-10
            @test norm(store.grad - G(test_point)) ≈ 0 atol = 1e-10

            ## objgrad
            output = OptimizationMethods.objgrad!(progData, precomp, store, 
                test_point)
            @test length(output) == 1
            @test output[1] ≈ F(test_point)
            @test norm(store.grad - G(test_point)) ≈ 0 atol = 1e-10

            ## hess
            OptimizationMethods.hess!(progData, precomp, store, test_point)
            @test H(test_point) ≈ store.hess
        end
    end

end

end