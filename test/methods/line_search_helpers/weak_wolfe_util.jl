# Date: 02/06/2025
# Author: Christian Varner
# Purpose: Test cases for wolfe ebls procedure

module TestEBLS

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Line Search Helper: EBLS" begin

# set seed for reproducibility
Random.seed!(1010)

###############################################################################
# Test definitions
###############################################################################

@test isdefined(OptimizationMethods, :EBLS!)

###############################################################################
# Test functionality implementation 1
###############################################################################

step_sizes = [.25, .5, 1., 2.]
δs = [2., 5., 10.]
c1s = [1e-6, 1e-4, 1e-2, 1e-1]
c2s = [1e-5, 1e-3, 1e-1, .9]

let αs = step_sizes, δs = δs, c1s = c1s, c2s = c2s

    # test problem
    progData = OptimizationMethods.LeastSquares(Float64)
    precomp, store = OptimizationMethods.initialize(progData)

    ############################################################################
    # Test cases group 1 -- wolfe should satisfy these conditions
    ############################################################################
    for α in αs
        for δ in δs
            for c1 in c1s
                for c2 in c2s

                    # get data for test
                    θk = zeros(50)
                    θkm1 = randn(50)
                    fkm1 = OptimizationMethods.obj(progData, θkm1)
                    gkm1 = OptimizationMethods.grad(progData, θkm1)
                    hkm1 = OptimizationMethods.hess(progData, θkm1)

                    # simulate a hessian step
                    step_direction = hkm1 \ gkm1

                    success = OptimizationMethods.EBLS!(
                        θk,
                        θkm1,
                        progData,
                        precomp,
                        store,
                        gkm1,
                        step_direction,
                        fkm1,
                        α,
                        δ,
                        c1,
                        c2;
                        max_iterations = 100
                    )

                    # make sure wolfe conditions are successful
                    @test success == true
                    
                    # check the conditions
                    αkm1 = -(θk - θkm1)[1]/step_direction[1]
                    dkm1 = (θk - θkm1)./αkm1
                    
                    ## correct steps are taken
                    @test dkm1 ≈ -step_direction

                    ## sufficient descent
                    fk = OptimizationMethods.obj(progData, θk)
                    @test fk <= fkm1 - c1 * αkm1 * dot(gkm1, step_direction)

                    ## curvature condition
                    gk = OptimizationMethods.grad(progData, θk)
                    @test dot(gk, -step_direction) >= c2 * dot(gkm1, -step_direction)
                end
            end
        end
    end
    ############################################################################

    ############################################################################
    # Test cases group 1 -- wolfe will not satisfy these conditions
    # (essential a test on the return flag)
    ############################################################################
    for α in αs
        for δ in δs
            for c1 in c1s
                for c2 in c2s

                    θk = zeros(50)
                    θkm1 = randn(50)
                    fkm1 = OptimizationMethods.obj(progData, θkm1)
                    gkm1 = OptimizationMethods.grad(progData, θkm1)

                    success = OptimizationMethods.EBLS!(
                        θk,
                        θkm1,
                        progData,
                        precomp,
                        store,
                        gkm1,
                        norm(gkm1)^2,
                        fkm1,
                        α,
                        δ,
                        c1,
                        c2;
                        max_iterations = 0
                    )

                    # make sure wolfe conditions are successful
                    @test success == false
                end
            end
        end
    end
    ############################################################################
end

###############################################################################
# Test functionality implementation 2
###############################################################################

step_sizes = [.25, .5, 1., 2.]
δs = [2., 5., 10.]
c1s = [1e-6, 1e-4, 1e-2, 1e-1]
c2s = [1e-5, 1e-3, 1e-1, .9]

let αs = step_sizes, δs = δs, c1s = c1s, c2s = c2s

    # test problem
    progData = OptimizationMethods.LeastSquares(Float64)
    precomp, store = OptimizationMethods.initialize(progData)

    ############################################################################
    # Test cases group 1 -- wolfe should satisfy these conditions
    ############################################################################
    for α in αs
        for δ in δs
            for c1 in c1s
                for c2 in c2s

                    θk = zeros(50)
                    θkm1 = randn(50)
                    fkm1 = OptimizationMethods.obj(progData, θkm1)
                    gkm1 = OptimizationMethods.grad(progData, θkm1)

                    success = OptimizationMethods.EBLS!(
                        θk,
                        θkm1,
                        progData,
                        precomp,
                        store,
                        gkm1,
                        norm(gkm1)^2,
                        fkm1,
                        α,
                        δ,
                        c1,
                        c2;
                        max_iterations = 100
                    )

                    # make sure wolfe conditions are successful
                    @test success == true
                    
                    # check the conditions
                    αkm1 = -(θk - θkm1)[1]/gkm1[1]
                    dkm1 = (θk - θkm1)./αkm1
                    
                    ## correct steps are taken
                    @test dkm1 ≈ -gkm1

                    ## sufficient descent
                    fk = OptimizationMethods.obj(progData, θk)
                    @test fk <= fkm1 - c1 * αkm1 * norm(gkm1)^2

                    ## curvature condition
                    gk = OptimizationMethods.grad(progData, θk)
                    @test dot(gk, -gkm1) >= c2 * dot(gkm1, -gkm1)
                end
            end
        end
    end
    ############################################################################

    ############################################################################
    # Test cases group 2 -- wolfe will not satisfy these conditions
    # (essential a test on the return flag)
    ############################################################################
    for α in αs
        for δ in δs
            for c1 in c1s
                for c2 in c2s

                    θk = zeros(50)
                    θkm1 = randn(50)
                    fkm1 = OptimizationMethods.obj(progData, θkm1)
                    gkm1 = OptimizationMethods.grad(progData, θkm1)

                    success = OptimizationMethods.EBLS!(
                        θk,
                        θkm1,
                        progData,
                        precomp,
                        store,
                        gkm1,
                        norm(gkm1)^2,
                        fkm1,
                        α,
                        δ,
                        c1,
                        c2;
                        max_iterations = 0
                    )

                    # make sure wolfe conditions are successful
                    @test success == false
                end
            end
        end
    end
    ############################################################################
end

end

end