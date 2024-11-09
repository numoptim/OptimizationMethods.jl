# OptimizationMethods.jl

module TestGDLipschitzApproximation

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Method: GD Lipschitz Approximation" begin

    #Testing Context 
    Random.seed!(1010)

    ##########################################
    # Test struct properties
    ##########################################

    ## test if method struct is defined 
    @test @isdefined LipschitzApproxGD
    
    ## test supertype of method struct 
    @test supertype(LipschitzApproxGD) == 
        OptimizationMethods.AbstractOptimizerData

    ## Test Field Names and Types 
    field_info(type::T) where T = [
        [:name, String], 
        [:init_stepsize, type], 
        [:threshold, type], 
        [:max_iterations, Int64],
        [:iter_diff, Vector{type}],
        [:grad_diff, Vector{type}],
        [:iter_hist, Vector{Vector{type}}],
        [:grad_val_hist, Vector{type}],
        [:stop_iteration, Int64]
    ]

    float_types = [Float16, Float32, Float64]
    param_dim = 10

    for float_type in float_types
        fields = field_info(float_type)
        optData = LipschitzApproxGD(
            float_type,
            x0 = zeros(float_type, param_dim), 
            init_stepsize = float_type(1e-4), 
            threshold = float_type(1e-4),
            max_iterations = 100
        )

        for field_elem in fields
            #Test Names (does not depend on field_type)
            @test field_elem[1] in fieldnames(LipschitzApproxGD)

            #Test Types
            @test field_elem[2] == typeof(getfield(optData, field_elem[1]))
        end
    end

    ## Test Assertions
    @test_throws AssertionError LipschitzApproxGD(
        Float64,
        x0 = randn(param_dim),
        init_stepsize = 0.0, 
        threshold = 1e-4,
        max_iterations = 100
    )

    ##########################################
    # Test optimizer
    ##########################################

    progData = OptimizationMethods.GaussianLeastSquares(Float64)
    x0 = progData.meta.x0
    ss = 1e-2

    ## One Step Update 
    optData = LipschitzApproxGD(
        Float64,
        x0 = x0,
        init_stepsize = ss,
        threshold = 1e-10,
        max_iterations = 1
    )

    x1 = lipschitz_approximation_gd(optData, progData)

    ### Test updated values 
    g0 = OptimizationMethods.grad(progData, x0)
    g1 = OptimizationMethods.grad(progData, x1)
    @test x1 ≈ x0 - ss * g0

    ### Test Stored values
    @test optData.prev_stepsize == ss
    @test optData.theta == Inf64
    @test optData.lipschitz_approximation ≈ norm(g1 - g0) / norm(x1 - x0)
    @test optData.iter_diff ≈ x1 - x0
    @test optData.grad_diff ≈ g1 - g0
    @test optData.iter_hist[1] == x0
    @test optData.iter_hist[2] == x1
    @test optData.grad_val_hist[1] ≈ norm(g0)
    @test optData.grad_val_hist[2] ≈ norm(g1)
    @test optData.stop_iteration == 1


    ## Two Step Update 
    optData = LipschitzApproxGD(
        Float64,
        x0 = x0,
        init_stepsize = ss,
        threshold = 1e-10,
        max_iterations = 2
    )

    x2 = lipschitz_approximation_gd(optData, progData)
    
    ### Test updated values 
    g0 = OptimizationMethods.grad(progData, x0)
    x1 = optData.iter_hist[2]
    g1 = OptimizationMethods.grad(progData, x1)
    g2 = OptimizationMethods.grad(progData, x2)
    @test x2 ≈ x1 - (norm(x1 - x0))/(2*norm(g1-g0)) * g1

    ### Test Stored values
    @test optData.prev_stepsize ≈ (norm(x1 - x0))/(2*norm(g1-g0))
    @test optData.theta ≈ optData.prev_stepsize / ss
    @test optData.lipschitz_approximation ≈ norm(g2 - g1) / norm(x2 - x1)
    @test optData.iter_diff ≈ x2 - x1
    @test optData.grad_diff ≈ g2 - g1
    @test optData.iter_hist[3] == x2 
    @test optData.grad_val_hist[2] ≈ norm(g1)
    @test optData.grad_val_hist[3] ≈ norm(g2)
    @test optData.stop_iteration == 2


    ## K Step Update K == 52
    optData = LipschitzApproxGD(
        Float64,
        x0 = x0,
        init_stepsize = ss,
        threshold = 1e-10,
        max_iterations = 100
    )

    xk = lipschitz_approximation_gd(optData, progData)

    ### Test updated values
    k = optData.stop_iteration
    xkm1 = optData.iter_hist[k-1]
    xkm2 = optData.iter_hist[k-2]
    xkm3 = optData.iter_hist[k-3]
   
    gkm3 = OptimizationMethods.grad(progData, xkm3)
    gkm2 = OptimizationMethods.grad(progData, xkm2)
    gkm1 = OptimizationMethods.grad(progData, xkm1)
    gk = OptimizationMethods.grad(progData, xk)

    λkm3 = - (xkm2[1] - xkm3[1]) / gkm3[1] 
    λkm2 = - (xkm1[1] - xkm2[1]) / gkm2[1]
    θkm2 = λkm3 / λkm2 
    λkm1 = min( sqrt(1 + θkm2)*λkm2, norm(xkm1 - xkm2)/(2*norm(gkm1 - gkm2)))
    @test xk ≈ xkm1 - λkm1 * gkm1

    ### Test stored values 
    @test optData.prev_stepsize ≈ λkm1 atol=1e-3
    @test optData.theta ≈ λkm1 / λkm2 atol=1e-3
    @test optData.lipschitz_approximation ≈ norm(gk - gkm1) / 
        norm(xk - xkm1) atol=1
    @test optData.iter_diff ≈ xk - xkm1 atol=1e-10
    @test optData.grad_diff ≈ gk - gkm1 atol=1e-9
    @test optData.grad_val_hist[k-1] ≈ norm(gkm1) atol=1e-9
    @test optData.grad_val_hist[k] ≈ norm(gk) atol=1e-9
end
end
