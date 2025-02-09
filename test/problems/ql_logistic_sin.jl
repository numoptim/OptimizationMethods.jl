# Date: 12/27/2021
# Author: Christian Varner
# Purpose: Implement test cases for quasi-likelihood objective functions

module TestQLLogisticSin

using Test, ForwardDiff, OptimizationMethods, Random, LinearAlgebra

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
    let real_types = [Float16, Float32, Float64], nobs_default = 1000,
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
    end

    # Test User-supplied Data Constructor
    let real_types = [Float16, Float32, Float64], nobs_default = 1000, 
        nobs_error = 900, nvar_default = 50, nvar_error = 45

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
    end

    ####################################
    # Test Struct: PrecomputeQLLogisticSin
    ####################################

    # Check if is defined
    @test isdefined(OptimizationMethods, :PrecomputeQLLogisticSin)
    
    # Test Super Type 
    @test supertype(OptimizationMethods.PrecomputeQLLogisticSin) ==
        OptimizationMethods.AbstractDefaultQLPrecompute

    # Test Fields 
    @test :obs_obs_t in fieldnames(OptimizationMethods.PrecomputeQLLogisticSin)

    # Test Constructor 
    let real_types = [Float16, Float32, Float64], nobs_default = 1000, 
        nvar_default = 50 

        for real_type in real_types 
            # Generate Random Problem 
            progData = OptimizationMethods.QLLogisticSin(real_type)

            # Generate Precompute 
            precomp = OptimizationMethods.PrecomputeQLLogisticSin(progData)

            # Check Field Type and Dimensions 
            @test typeof(precomp.obs_obs_t) == Array{real_type, 3}
            @test size(precomp.obs_obs_t) == (nobs_default, nvar_default, nvar_default)

            # Compare Values 
            @test reduce(+, precomp.obs_obs_t, dims=1)[1,:,:] ≈
                progData.design'*progData.design atol=
                eps(real_type) *nobs_default * nvar_default
        end
    end

    ####################################
    # Test Struct: Allocation
    ####################################

    # Check if struct is defined 
    @test isdefined(OptimizationMethods, :AllocateQLLogisticSin)

    # Test super type 
    @test supertype(OptimizationMethods.AllocateQLLogisticSin) == 
        OptimizationMethods.AbstractDefaultQLAllocate

    # Test Fields 
    for name in [:linear_effect, :μ, :∇μ_η, :∇∇μ_η, :variance, :∇variance, 
        :weighted_residual, :grad, :hess]
        @test name in fieldnames(OptimizationMethods.AllocateQLLogisticSin)
    end

    # Test Constructors 
    let real_types = [Float16, Float32, Float64], nobs_default = 1000 , 
        nvar_default = 50 

        for real_type in real_types 
            # Generate Random Problem 
            progData = OptimizationMethods.QLLogisticSin(real_type)

            # Generate Store 
            store = OptimizationMethods.AllocateQLLogisticSin(progData)

            # Check field Type and Dimensions
            fields_nametypesize = [
                [:linear_effect, Vector{real_type}, nobs_default],
                [:μ, Vector{real_type}, nobs_default],
                [:∇μ_η, Vector{real_type}, nobs_default],
                [:∇∇μ_η, Vector{real_type}, nobs_default],
                [:variance, Vector{real_type}, nobs_default],
                [:∇variance, Vector{real_type}, nobs_default],
                [:weighted_residual, Vector{real_type}, nobs_default],
                [:grad, Vector{real_type}, nvar_default],
                [:hess, Matrix{real_type}, (nvar_default, nvar_default)]
            ]

            for f_nts in fields_nametypesize
                fld = getfield(store, f_nts[1])
                @test typeof(fld) == f_nts[2]
                @test length(f_nts[3]) == 1 ? (length(fld) == f_nts[3]) : 
                    (size(fld) == f_nts[3])
            end
            
        end
    end

    ####################################
    # Test Method: Initialize
    ####################################
    let real_types = [Float16, Float32, Float64], nobs_default = 1000,
        nvar_default = 50 

        for real_type in real_types 
            progData = OptimizationMethods.QLLogisticSin(real_type)
            precomp, store = OptimizationMethods.initialize(progData)

            @test typeof(precomp) == 
                OptimizationMethods.PrecomputeQLLogisticSin{real_type}
            @test typeof(store) == 
                OptimizationMethods.AllocateQLLogisticSin{real_type}
        end

    end 

    ####################################
    # Test Methods 
    ####################################
    let real_types = [Float32, Float64], nobs_default = 1000,
        nvar_default = 50, nargs=1

        for real_type in real_types
            progData = OptimizationMethods.QLLogisticSin(real_type)
            precomp, store = OptimizationMethods.initialize(progData)
            arg_tests = [randn(real_type, nvar_default) / real_type(sqrt(nvar_default))
                for i = 1:nargs]

            nevals_obj = 1
            nevals_grad = 1 
            nevals_hess = 1

            function gradient(x)
                linear_effect = progData.design * x 
                predicted = 1 ./ (1 .+ exp.(-linear_effect)) 
                variance = 1 .+ predicted + sin.(2*pi*predicted)
                residual = (progData.response - predicted) ./ variance 
                d_predicted = predicted .* (1 .- predicted)
                return  - progData.design'*(residual .* d_predicted)
            end

            ####################################
            # Test Methods: Gradient Evaluation 
            ####################################
            for x in arg_tests
                g = gradient(x) 

                # Without Precompute 
                @test g ≈ OptimizationMethods.grad(progData, x)
                @test progData.counters.neval_grad == nevals_grad 
                nevals_grad += 1

                # With Precompute 
                @test g ≈ OptimizationMethods.grad(progData, precomp, x)
                @test progData.counters.neval_grad == nevals_grad
                nevals_grad += 1 

                # With Storage
                OptimizationMethods.grad!(progData, precomp, store, x)
                @test g ≈ store.grad
                @test progData.counters.neval_grad == nevals_grad 
                nevals_grad += 1
            end

            ####################################
            # Test Methods: Hessian Evaluation 
            ####################################
            for x in arg_tests
                h = ForwardDiff.jacobian(gradient, x)

                # Without Precompute 
                @test h ≈ OptimizationMethods.hess(progData, x)
                @test progData.counters.neval_hess == nevals_hess 
                nevals_hess += 1 

                # With Precomputation 
                @test h ≈ OptimizationMethods.hess(progData, precomp, x)
                @test progData.counters.neval_hess == nevals_hess 
                nevals_hess += 1

                # With Preallocation 
                OptimizationMethods.hess!(progData, precomp, store, x)
                @test h ≈ store.hess 
                @test progData.counters.neval_hess == nevals_hess 
                nevals_hess += 1
            end

            ####################################
            # Test Methods: Objective Evaluation 
            ####################################
            baseline_arg = zeros(real_type, nvar_default)
            
            # Objective Difference Trapezoidal Rule
            function trapezoidal(y, x=baseline_arg)
                Δ = 1e-3
                δ = y - x
                interpolation_points = 0:Δ:1.0
                obj_diff_approx = 0.0 
                for i in 1:(length(interpolation_points)-1)
                    t = interpolation_points[i]
                    t_next = interpolation_points[i+1]
                    obj_diff_approx += (Δ/2)*(
                        dot(gradient(x + t*δ), δ) + 
                        dot(gradient(x + t_next*δ), δ)
                    )
                end

                return obj_diff_approx 
            end

            # Baseline Evaluation
            obj_base = OptimizationMethods.obj(progData, baseline_arg) 
            nevals_obj += 1

            for x in arg_tests
                obj_diff = trapezoidal(x)

                # Without Precompute 
                @test obj_diff ≈ OptimizationMethods.obj(progData, x) - obj_base
                @test progData.counters.neval_obj == nevals_obj 
                nevals_obj += 1

                # With Precompute 
                @test obj_diff ≈ OptimizationMethods.obj(progData, precomp, x) - 
                    obj_base
                @test progData.counters.neval_obj == nevals_obj 
                nevals_obj += 1

                 # With Store 
                 @test obj_diff ≈ OptimizationMethods.obj(progData, precomp, store, x) - 
                    obj_base
                 @test progData.counters.neval_obj == nevals_obj 
                 nevals_obj += 1
            end

            ####################################
            # Test Methods: Objective-Gradient Evaluation 
            ####################################

            # Same baseline objective function for comparison 
            # Same trapezoidal rule for approximate objective calculation 
            
            for x in arg_tests
                obj_diff = trapezoidal(x)
                gra = gradient(x)

                # Without Precompute 
                o, g = OptimizationMethods.objgrad(progData, x)
                @test o - obj_base ≈ obj_diff 
                @test g ≈ gra 
                @test progData.counters.neval_obj == nevals_obj 
                @test progData.counters.neval_grad == nevals_grad 
                nevals_obj += 1
                nevals_grad += 1

                # With Precomputation 
                o, g = OptimizationMethods.objgrad(progData, precomp, x)
                @test o - obj_base ≈ obj_diff 
                @test g ≈ gra
                @test progData.counters.neval_obj == nevals_obj 
                @test progData.counters.neval_grad == nevals_grad 
                nevals_obj += 1
                nevals_grad += 1

                # With Store 
                o = OptimizationMethods.objgrad!(progData, precomp, store, x)
                @test o - obj_base ≈ obj_diff
                @test store.grad ≈ gra
                @test progData.counters.neval_obj == nevals_obj 
                @test progData.counters.neval_grad == nevals_grad 
                nevals_obj += 1
                nevals_grad += 1
            end
        end
    end

end

end