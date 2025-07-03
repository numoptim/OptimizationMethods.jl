# Date: 12/27/2021
# Author: Christian Varner
# Purpose: Implement test cases for quasi-likelihood objective functions

module TestQuasiLikelihood

using Test, ForwardDiff, OptimizationMethods, Random, LinearAlgebra

################################################################################
# Testing Functions
################################################################################

function test_methods(
    progData::P where P <: OptimizationMethods.AbstractDefaultQL{T, S},
    gradient::Function,
    nvar_default::Int64,
    nargs::Int64
    ) where {T, S}

    precomp, store = OptimizationMethods.initialize(progData)
    arg_tests = [randn(T, nvar_default) / T(sqrt(nvar_default))
        for i = 1:nargs]

    nevals_obj = 1
    nevals_grad = 1 
    nevals_hess = 1

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
    baseline_arg = zeros(T, nvar_default)
    
    # Objective Difference Trapezoidal Rule
    function trapezoidal(y, x=baseline_arg)
        Δ = 1e-4
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
        @test obj_diff ≈ OptimizationMethods.obj(progData, x) - obj_base rtol = 1e-5
        @test progData.counters.neval_obj == nevals_obj 
        nevals_obj += 1

        # With Precompute 
        @test obj_diff ≈ OptimizationMethods.obj(progData, precomp, x) - 
            obj_base rtol = 1e-5
        @test progData.counters.neval_obj == nevals_obj 
        nevals_obj += 1

        # With Store 
        @test obj_diff ≈ OptimizationMethods.obj(progData, precomp, store, x) - 
        obj_base rtol = 1e-5
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
        @test o - obj_base ≈ obj_diff rtol = 1e-5
        @test g ≈ gra 
        @test progData.counters.neval_obj == nevals_obj 
        @test progData.counters.neval_grad == nevals_grad 
        nevals_obj += 1
        nevals_grad += 1

        # With Precomputation 
        o, g = OptimizationMethods.objgrad(progData, precomp, x)
        @test o - obj_base ≈ obj_diff rtol = 1e-5
        @test g ≈ gra
        @test progData.counters.neval_obj == nevals_obj 
        @test progData.counters.neval_grad == nevals_grad 
        nevals_obj += 1
        nevals_grad += 1

        # With Store 
        o = OptimizationMethods.objgrad!(progData, precomp, store, x)
        @test o - obj_base ≈ obj_diff rtol = 1e-5
        @test store.grad ≈ gra
        @test progData.counters.neval_obj == nevals_obj 
        @test progData.counters.neval_grad == nevals_grad 
        nevals_obj += 1
        nevals_grad += 1
    end
end

################################################################################
# Gradient functions for testing QL functionality
################################################################################

function gradient_logistic_sin(
    x, 
    progData::P where P <: OptimizationMethods.AbstractDefaultQL{T, S}
    ) where {T, S}

    linear_effect = progData.design * x 
    predicted = 1 ./ (1 .+ exp.(-linear_effect)) 
    variance = 1 .+ predicted + sin.(2*pi*predicted)
    residual = (progData.response - predicted) ./ variance 
    d_predicted = predicted .* (1 .- predicted)

    return  - progData.design'*(residual .* d_predicted)
end

function gradient_logistic_monomial(
    x,
    progData::P where P <: OptimizationMethods.AbstractDefaultQL{T, S}
) where {T, S}

    p = progData.p
    c = progData.c

    η = progData.design * x
    μ = 1 ./ (1 .+ exp.(-η))
    dμ = μ .* (1 .- μ)
    V = abs.(μ).^(2*p) .+ c

    return -progData.design' * (dμ .* (progData.response .- μ) ./ V)
end

function gradient_logistic_centered_log(
    x,
    progData::P where P <: OptimizationMethods.AbstractDefaultQL{T, S}
) where {T, S}

    p = progData.p
    c = progData.c
    d = progData.d

    η = progData.design * x
    μ = 1 ./ (1 .+ exp.(-η))
    dμ = μ .* (1 .- μ)
    V = log.(abs.(μ.-c).^(2p).+1) .+ d

    return -progData.design' * (dμ .* (progData.response .- μ) ./ V)
end

function gradient_logistic_centered_exp(
    x,
    progData::P where P <: OptimizationMethods.AbstractDefaultQL{T, S}
) where {T, S}

    c = progData.c
    p = progData.p

    η = progData.design * x
    μ = 1 ./ (1 .+ exp.(-η))
    dμ = μ .* (1 .- μ)
    V = exp.(-abs.(μ.-c).^(2*p))

    return -progData.design'*(dμ .* (progData.response .- μ)./V)
end

################################################################################
# Testing set for Quasi-likelihood problems
################################################################################

const ql_structures = [OptimizationMethods.QLLogisticSin, 
    OptimizationMethods.QLLogisticMonomial,
    OptimizationMethods.QLLogisticCenteredExp,
    OptimizationMethods.QLLogisticCenteredLog]
const ql_structure_symbols = [:QLLogisticSin, :QLLogisticMonomial, 
    :QLLogisticCenteredExp, :QLLogisticCenteredLog]
const ql_gradients = [gradient_logistic_sin, gradient_logistic_monomial, 
    gradient_logistic_centered_exp, gradient_logistic_centered_log]

const ql_precomp_types = [OptimizationMethods.PrecomputeQLLogisticSin,
    OptimizationMethods.PrecomputeQLLogisticMonomial,
    OptimizationMethods.PrecomputeQLLogisticCenteredExp,
    OptimizationMethods.PrecomputeQLLogisticCenteredLog]
const ql_precomp_symbols = [:PrecomputeQLLogisticSin, 
    :PrecomputeQLLogisticMonomial,
    :PrecomputeQLLogisticCenteredExp,
    :PrecomputeQLLogisticCenteredLog]

const ql_allocate_types = [OptimizationMethods.AllocateQLLogisticSin,
    OptimizationMethods.AllocateQLLogisticMonomial,
    OptimizationMethods.AllocateQLLogisticCenteredExp,
    OptimizationMethods.AllocateQLLogisticCenteredLog] 
const ql_allocate_symbols = [:AllocateQLLogisticSin, :AllocateQLLogisticMonomial, 
    :AllocateQLLogisticCenteredExp, :AllocateQLLogisticCenteredLog]

@testset "Quasi-likelihood Problems" begin

    ####################################
    # Test Struct: Precompute structures
    ####################################

    # Test Constructor 
    let real_types = [Float16, Float32, Float64], 
        nobs_default = 1000, 
        nvar_default = 50,
        structures = ql_structures
        precomp_types = ql_precomp_types
        precomp_symbols = ql_precomp_symbols

        # check definitions
        for sym in precomp_symbols
            @test isdefined(OptimizationMethods, sym)
        end

        # check super types
        for precomp_type in precomp_types
            @test supertype(precomp_type) == 
                OptimizationMethods.AbstractDefaultQLPrecompute
        end

        # test fields
        for precomp_type in precomp_types
            @test :obs_obs_t in fieldnames(precomp_type)
        end

        for real_type in real_types 
            for (ql_struct, precomp_type) in zip(structures, precomp_types)
                # Generate Random Problem 
                progData = ql_struct(real_type)

                # Generate Precompute 
                precomp = precomp_type(progData)

                # Check Field Type and Dimensions 
                @test typeof(precomp.obs_obs_t) == Array{real_type, 3}
                @test size(precomp.obs_obs_t) == (nobs_default, nvar_default, nvar_default)

                # Compare Values 
                @test reduce(+, precomp.obs_obs_t, dims=1)[1,:,:] ≈
                    progData.design'*progData.design atol=
                    eps(real_type) *nobs_default * nvar_default
            end
        end
    end

    ####################################
    # Test Struct: Allocation
    ####################################

    # Test Constructors 
    let real_types = [Float16, Float32, Float64], 
        nobs_default = 1000, 
        nvar_default = 50,
        structures = ql_structures,
        allocate_types = ql_allocate_types
        allocate_symbols = ql_allocate_symbols

        # Check if structures are well-defined
        for sym in allocate_symbols
            @test isdefined(OptimizationMethods, sym)
        end

        # test super type for each allocate type
        for allocate_type in allocate_types
            @test supertype(allocate_type) == 
                OptimizationMethods.AbstractDefaultQLAllocate
        end

        # test field for each allocate type
        for allocate_type in allocate_types
            for name in [:linear_effect, :μ, :∇μ_η, :∇∇μ_η, :variance, :∇variance, 
                :weighted_residual, :grad, :hess]
                @test name in fieldnames(allocate_type)
            end
        end

        for real_type in real_types 
            for (ql_struct, allocate_type) in zip(structures, allocate_types)

                # Generate Random Problem 
                progData = ql_struct(real_type)

                # Generate Store 
                store = allocate_type(progData)

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
    end

    ####################################
    # Test Method: Initialize
    ####################################
    let real_types = [Float16, Float32, Float64], 
        nobs_default = 1000,
        nvar_default = 50, 
        structures = ql_structures,
        precomp_types = ql_precomp_types,
        allocate_types = ql_allocate_types 

        for real_type in real_types 
            for (ql_struct, precomp_type, allocate_type) in 
                zip(structures, precomp_types, allocate_types)

                progData = ql_struct(real_type)
                precomp, store = OptimizationMethods.initialize(progData)

                @test typeof(precomp) == precomp_type{real_type}
                @test typeof(store) == allocate_type{real_type}
            end
        end
    end 

    ####################################
    # Test Methods -- Quasi-Likelihood
    ####################################
    let real_types = [Float64], 
        nobs_default = 1000,
        nvar_default = 50, nargs=1,
        structures = ql_structures
        gradients = ql_gradients

        for real_type in real_types
            for (ql_struct, ql_gradient) in zip(structures, gradients)
                progData = ql_struct(real_type)

                gradient(x) = ql_gradient(x, progData)
                test_methods(
                    progData,
                    gradient,
                    nvar_default,
                    nargs
                )
            end
        end
    end
end

end