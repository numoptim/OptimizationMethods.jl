# Date: 10/29/2024
# Author: Christian Varner
# Purpose: Test the implementation of Gradient Descent with Fixed 
# step size found in src/methods/gd-fixed.jl

module TestGDFixed

using Test, OptimizationMethods, Random, LinearAlgebra

@testset "Method: GD Fixed" begin

    # set the seed
    Random.seed!(1010)

    ##########################################
    # Test struct properties
    ##########################################

    # test is defined
    @test @isdefined FixedStepGD

    # test super typeof
    @test supertype(FixedStepGD) == OptimizationMethods.AbstractOptimizerData

    # test field names
    @test :name in fieldnames(FixedStepGD)
    @test :step_size in fieldnames(FixedStepGD)
    @test :threshold in fieldnames(FixedStepGD)
    @test :max_iterations in fieldnames(FixedStepGD)
    @test :iter_hist in fieldnames(FixedStepGD)
    @test :grad_val_hist in fieldnames(FixedStepGD)
    @test :stop_iteration in fieldnames(FixedStepGD)
   
    ##########################################
    # End Test struct properties
    ##########################################

    ##########################################
    # Test struct constructor
    ##########################################

    # random arguments -- dimension larger than 1
    T = Int32
    x0 = zeros(T, 10)
    step_size = T(rand(collect(1:1000)))
    max_iterations = rand(collect(1:1000)) 
    solver = FixedStepGD(T,
        x0 = x0,
        step_size = step_size,
        threshold = T(1),
        max_iterations = max_iterations
        )

    # test types
    @test typeof(solver.step_size) == T
    @test typeof(solver.threshold) == T
    @test typeof(solver.max_iterations) == Int64
    @test typeof(solver.iter_hist) == Vector{Vector{T}}
    @test typeof(solver.grad_val_hist) == Vector{T}
    @test typeof(solver.stop_iteration) == Int64

    # test values
    @test solver.name == "Gradient Descent with Fixed Step Size" 
    @test solver.step_size == step_size
    @test solver.threshold == 1
    @test solver.max_iterations == max_iterations
    @test size(solver.iter_hist) == (max_iterations+1, ) 

    flag = true
    for i in 1:max_iterations+1
        flag = flag && (size(solver.iter_hist[i]) == (10,))
    end
    @test flag

    @test solver.iter_hist[1] == x0
    @test size(solver.grad_val_hist) == (max_iterations+1, )
    @test solver.stop_iteration == -1


    # random arguments -- dimension 1
    T = Int32
    x0 = zeros(T, 1)
    step_size = T(rand(collect(1:1000)))
    max_iterations = rand(collect(1:1000)) 
    solver = FixedStepGD(T,
        x0 = x0,
        step_size = step_size,
        threshold = T(1),
        max_iterations = max_iterations
        )

    # test types
    @test typeof(solver.step_size) == T
    @test typeof(solver.threshold) == T
    @test typeof(solver.max_iterations) == Int64
    @test typeof(solver.iter_hist) == Vector{Vector{T}}
    @test typeof(solver.grad_val_hist) == Vector{T}
    @test typeof(solver.stop_iteration) == Int64

    # test values
    @test solver.name == "Gradient Descent with Fixed Step Size" 
    @test solver.step_size == step_size
    @test solver.threshold == 1
    @test solver.max_iterations == max_iterations
    @test size(solver.iter_hist) == (max_iterations+1, ) 

    flag = true
    for i in 1:max_iterations+1
        flag = flag && (size(solver.iter_hist[i]) == (1,))
    end
    @test flag

    @test solver.iter_hist[1] == x0
    @test size(solver.grad_val_hist) == (max_iterations+1, )
    @test solver.stop_iteration == -1

    ##########################################
    # End test struct constructor
    ##########################################

    ##########################################
    # Test optimizer
    ##########################################

    progData = OptimizationMethods.LeastSquares(Float64)
    x0 = randn(50)
    optData = OptimizationMethods.FixedStepGD(
        Float64, 
        x0=x0, 
        step_size=0.0005, 
        threshold=1e-10, 
        max_iterations=1
    )

    # test first iteration
    x1 = fixed_step_gd(optData, progData)

    ## test that the correct step was applied
    @test x0 - x1 ≈ 0.0005 * OptimizationMethods.grad(progData, x0) 

    ## test the struct values
    @test optData.iter_hist[1] == x0
    @test optData.iter_hist[2] == x1
    @test optData.grad_val_hist[1] ≈ norm(OptimizationMethods.grad(progData, x0))
    @test optData.grad_val_hist[2] ≈ norm(OptimizationMethods.grad(progData, x1))
    @test optData.stop_iteration == 1

    ## test random iteration and random step size and random threshold
    k = rand(collect(3:10))
    step_size = abs(randn(1)[1])
    optData = OptimizationMethods.FixedStepGD(
        Float64,
        x0 = x0,
        step_size = step_size,
        threshold = 1e-5*abs(randn(1)[1]),
        max_iterations = k 
    )
    
    xk = fixed_step_gd(optData, progData)  

    ## test that the struct values are correct and that the iteration is correct
    for iter in 1:k
        x0 .-= step_size * OptimizationMethods.grad(progData, x0) 
        @test optData.iter_hist[iter + 1] ≈ x0
        
        # this test can fail if iter_hist is not correctly saved OR
        # gra val hist is not saved correctly
        gi = OptimizationMethods.grad(progData, optData.iter_hist[iter + 1])
        @test optData.grad_val_hist[iter + 1] ≈ norm(gi)
    end

    @test optData.stop_iteration == k
    
    # Test that the gradient tolerance condition is triggered + iteration is counted correctly
    optData = OptimizationMethods.FixedStepGD(
        Float64,
        x0 = x0,
        step_size = 0.0005,
        threshold = 10.0,
        max_iterations = 1000 
    )
    xk = fixed_step_gd(optData, progData) 

    @test optData.grad_val_hist[optData.stop_iteration + 1] <= 10.0
    @test optData.grad_val_hist[optData.stop_iteration] > 10.0

    
    ##########################################
    # End Test optimizer
    ##########################################

end # end test set

end # end module
