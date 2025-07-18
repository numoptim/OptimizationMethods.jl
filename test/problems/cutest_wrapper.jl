# Date: 2025/7/18
# Author: Christian Varner
# Purpose: Test the implementation of CUTEst functionality

module TestCUTEstWrapper

using Test, OptimizationMethods, LinearAlgebra, Random, CUTEst, NLPModels

@testset "Test PrecomputeCUTEst Struct" begin

    # test definition
    @test isdefined(OptimizationMethods, :PrecomputeCUTEst)

    # test properties
    @test supertype(OptimizationMethods.PrecomputeCUTEst) ==
        OptimizationMethods.AbstractPrecompute
    @test length(fieldnames(OptimizationMethods.PrecomputeCUTEst)) == 0

    # test constructor
    progData = CUTEstModel{Float64}("ROSENBR")
    precomp = OptimizationMethods.PrecomputeCUTEst(progData)
    @test typeof(precomp) == OptimizationMethods.PrecomputeCUTEst{Float64}

    # close problem
    finalize(progData)

end

@testset "Test AllocateCUTEst Struct" begin

    # test definition
    @test isdefined(OptimizationMethods, :AllocateCUTEst)

    # test properties
    @test supertype(OptimizationMethods.AllocateCUTEst) ==
        OptimizationMethods.AbstractProblemAllocate
    @test :grad in fieldnames(OptimizationMethods.AllocateCUTEst)
    @test :hess in fieldnames(OptimizationMethods.AllocateCUTEst)
    
    # test constructor -- problem 1
    progData = CUTEstModel{Float64}("ROSENBR")
    store = OptimizationMethods.AllocateCUTEst(progData)

    nvar = progData.meta.nvar
    @test length(store.grad) == nvar
    @test size(store.hess) == (nvar, nvar)

    # close problem
    finalize(progData)

    # test constructor -- problem 2
    progData = CUTEstModel{Float64}("WOODS")
    store = OptimizationMethods.AllocateCUTEst(progData)

    nvar = progData.meta.nvar
    @test length(store.grad) == nvar
    @test size(store.hess) == (nvar, nvar)

    # close problem
    finalize(progData)

end

@testset "Test initialize(::CUTEstModel) Functionality" begin

    # test -- problem 1
    progData = CUTEstModel{Float64}("ROSENBR")
    returned = OptimizationMethods.initialize(progData)

    # test correct return
    @test length(returned) == 2
    @test typeof(returned[1]) == OptimizationMethods.PrecomputeCUTEst{Float64}
    @test typeof(returned[2]) == OptimizationMethods.AllocateCUTEst{Float64}
    
    # test store
    precomp, store = returned[1], returned[2]

    nvar = progData.meta.nvar
    @test length(store.grad) == nvar
    @test size(store.hess) == (nvar, nvar)

    finalize(progData)

    # test -- problem 2
    progData = CUTEstModel{Float64}("WOODS")
    returned = OptimizationMethods.initialize(progData)

    # test correct return
    @test length(returned) == 2
    @test typeof(returned[1]) == OptimizationMethods.PrecomputeCUTEst{Float64}
    @test typeof(returned[2]) == OptimizationMethods.AllocateCUTEst{Float64}
    
    # test store
    precomp, store = returned[1], returned[2]

    nvar = progData.meta.nvar
    @test length(store.grad) == nvar
    @test size(store.hess) == (nvar, nvar)

    finalize(progData)

end

@testset "Test CUTEstWrapper Functionality -- Group 1" begin

    # test problem 1 
    progData = CUTEstModel{Float64}("ROSENBR") 
    nvar = progData.meta.nvar

    # test output
    for i in 1:10

        # generate random point
        x = randn(nvar)

        # compute and test
        o = OptimizationMethods.obj(progData, x)
        @test o == NLPModels.obj(progData, x)

        # compute and test
        g = OptimizationMethods.grad(progData, x)

        # compute and test 
        o, g = OptimizationMethods.grad(progData, x) 

        # compute and test
        h = OptimizationMethods.hess(progData, x)

    end

    finalize(progData)

end

@testset "Test CUTEstWrapper Functionality -- Group 3" begin


end

end