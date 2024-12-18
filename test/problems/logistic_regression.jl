# Date: 10/05/2024
# Author: Christian Varner
# Purpose: Implement some test cases for logistic regression

module TestLogisticRegression

using Test, ForwardDiff, OptimizationMethods, Random, LinearAlgebra, NLPModels

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
    
    @test isdefined(OptimizationMethods, :PrecomputeLogReg)
    @test size(fieldnames(OptimizationMethods.PrecomputeLogReg), 1) == 0
    
    ####################################
    # Test Allocated struct
    ####################################
    
    # Test if it is defined and the field names
    @test isdefined(OptimizationMethods, :AllocateLogReg)
    
    field_names = [:linear_effect, :probabilities, :residuals, :grad, :hess]
    for name in field_names
        @test name in fieldnames(OptimizationMethods.AllocateLogReg)
    end

    # test constructor
    progDataConstructor = OptimizationMethods.LogisticRegression
    real_types = [Float16, Float32, Float64]
    for first_constructor in [true, false]
        for real_type in real_types
            progData = first_constructor ? progDataConstructor(real_type) :
            OptimizationMethods.LogisticRegression(Matrix{real_type}(randn(100, 10)), 
                                                       Vector{Bool}(bitrand(nobs)),
                                                       x0 = Vector{real_type}(randn(10)))
            store = OptimizationMethods.AllocateLogReg(progData)
            
            @test typeof(store.linear_effect) == Vector{real_type}
            @test typeof(store.probabilities) == Vector{real_type}
            @test typeof(store.residuals) == Vector{real_type}
            @test typeof(store.grad) == Vector{real_type}
            @test typeof(store.hess) == Matrix{real_type}

            @test size(store.linear_effect, 1) == size(progData.design, 1)
            @test size(store.probabilities, 1) == size(progData.design, 1)
            @test size(store.residuals, 1) == size(progData.design, 1)
            @test size(store.grad, 1) == size(progData.design, 2)
            @test size(store.hess) == (size(progData.design, 2), size(progData.design, 2))
        end
    end

    ####################################
    # Test initialize 
    ####################################

    progDataConstructor = OptimizationMethods.LogisticRegression
    real_types = [Float16, Float32, Float64]
    for first_constructor in [true, false]
        for real_type in real_types
            progData = first_constructor ? progDataConstructor(real_type) :
            OptimizationMethods.LogisticRegression(Matrix{real_type}(randn(100, 10)), 
                                                       Vector{Bool}(bitrand(nobs)),
                                                       x0 = Vector{real_type}(randn(10)))
            output = OptimizationMethods.initialize(progData)

            @test size(output, 1) == 2

            out1, out2 = output[1], output[2]
            @test typeof(out1) == OptimizationMethods.PrecomputeLogReg{real_type}
            @test typeof(out2) == OptimizationMethods.AllocateLogReg{real_type}
        end
    end

    ####################################
    # Test functionality - group 1
    ####################################

    # test problem 1
    p = 100
    nobs = p
    nvar = p

    A = Matrix{Float64}(Matrix(I, p, p))
    b = Vector{Bool}(bitrand(p))

    progData = OptimizationMethods.LogisticRegression(A, b)
    x = zeros(Float64, p)

    ## test objective
    o = obj(progData, x)
    @test isapprox(o, -p*log(.5); atol = 1e-10)
    @test progData.counters.neval_obj == 1
    @test progData.counters.neval_grad == 0
    @test progData.counters.neval_hess == 0

    ## test gradient
    g = grad(progData, x)
    @test isapprox(norm(g - ((.5) .* ones(p) - b)), 0, atol = 1e-10)
    @test progData.counters.neval_obj == 1
    @test progData.counters.neval_grad == 1
    @test progData.counters.neval_hess == 0

    ## test objgrad
    output = objgrad(progData, x)
    @test length(output) == 2

    o, g = output[1], output[2]
    @test isapprox(o, -p*log(.5); atol = 1e-10)
    @test isapprox(norm(g - ((.5) .* ones(p) - b)), 0, atol = 1e-10) 
    @test progData.counters.neval_obj == 2
    @test progData.counters.neval_grad == 2
    @test progData.counters.neval_hess == 0

    ## test hess
    trueHessian = .25 .* A
    h = OptimizationMethods.hess(progData, x)
    @test isapprox(norm(h - trueHessian), 0; atol = 1e-10)
    @test progData.counters.neval_obj == 2
    @test progData.counters.neval_grad == 2
    @test progData.counters.neval_hess == 1

    # test problem 2
    
    progData = OptimizationMethods.LogisticRegression(Float64)
    nobs, nvar = size(progData.design)
    A = progData.design
    b = progData.response
    logistic = OptimizationMethods.logistic

    function F(x::Vector)
        probs = logistic.(A * x)
        return - b' * log.(probs) - (1 .- b)' * log.(1 .- probs)
    end

    x = randn(nvar)

    ## test objective
    o = obj(progData, x)
    o_true = F(x)
    @test isapprox(o - o_true, 0, atol = 1e-10)

    ## test gradient
    g = grad(progData, x)
    g_autodiff = ForwardDiff.gradient(F, x) 
    @test isapprox(norm(g - g_autodiff), 0, atol = 1e-10)

    ## test objgrad
    output = objgrad(progData, x)
    o, g = output[1], output[2]
   
    g_autodiff = ForwardDiff.gradient(F, x)
    @test isapprox(norm(g - g_autodiff), 0, atol = 1e-10) 
    @test isapprox(norm(o - F(x)), 0, atol = 1e-10)

    ## test hess
    h = OptimizationMethods.hess(progData, x)
    h_autodiff = ForwardDiff.hessian(F, x)
    @test isapprox(norm(h - h_autodiff), 0; atol = 1e-10)

    ####################################
    # Test functionality - group 2
    ####################################
    
    # test problem 1
    p = 100
    nobs = p
    nvar = p

    A = Matrix{Float64}(Matrix(I, p, p))
    b = Vector{Bool}(bitrand(p))

    progData = OptimizationMethods.LogisticRegression(A, b)
    precomp, store = OptimizationMethods.initialize(progData)
    x = zeros(Float64, p)

    ## test objective
    o = obj(progData, precomp, x)
    @test isapprox(o, -p*log(.5); atol = 1e-10)
    @test progData.counters.neval_obj == 1
    @test progData.counters.neval_grad == 0
    @test progData.counters.neval_hess == 0

    ## test gradient
    g = grad(progData, precomp, x)
    @test isapprox(norm(g - ((.5) .* ones(p) - b)), 0, atol = 1e-10) 
    @test progData.counters.neval_obj == 1
    @test progData.counters.neval_grad == 1
    @test progData.counters.neval_hess == 0

    ## test objgrad
    output = objgrad(progData, precomp, x)
    @test length(output) == 2

    o, g = output[1], output[2]
    @test isapprox(o, -p*log(.5); atol = 1e-10)
    @test isapprox(norm(g - ((.5) .* ones(p) - b)), 0, atol = 1e-10) 
    @test progData.counters.neval_obj == 2
    @test progData.counters.neval_grad == 2
    @test progData.counters.neval_hess == 0

    ## test hess
    trueHessian = .25 .* A
    h = OptimizationMethods.hess(progData, precomp, x)
    @test isapprox(norm(h - trueHessian), 0; atol = 1e-10)
    @test progData.counters.neval_obj == 2
    @test progData.counters.neval_grad == 2
    @test progData.counters.neval_hess == 1
    
    # test problem 2

    progData = OptimizationMethods.LogisticRegression(Float64)
    precomp, store = OptimizationMethods.initialize(progData)
    nobs, nvar = size(progData.design)
    A = progData.design
    b = progData.response
    logistic = OptimizationMethods.logistic

    function F(x::Vector)
        probs = logistic.(A * x)
        return - b' * log.(probs) - (1 .- b)' * log.(1 .- probs)
    end

    x = randn(nvar)

    ## test objective
    o = obj(progData, precomp, x)
    o_true = F(x)
    @test isapprox(o - o_true, 0, atol = 1e-10)

    ## test gradient
    g = grad(progData, precomp, x)
    g_autodiff = ForwardDiff.gradient(F, x) 
    @test isapprox(norm(g - g_autodiff), 0, atol = 1e-10)

    ## test objgrad
    output = objgrad(progData, precomp, x)
    o, g = output[1], output[2]
   
    g_autodiff = ForwardDiff.gradient(F, x)
    @test isapprox(norm(g - g_autodiff), 0, atol = 1e-10) 
    @test isapprox(norm(o - F(x)), 0, atol = 1e-10)

    ## test hess
    h = OptimizationMethods.hess(progData, precomp, x)
    h_autodiff = ForwardDiff.hessian(F, x)
    @test isapprox(norm(h - h_autodiff), 0; atol = 1e-10)
    
    ####################################
    # Test functionality - group 3
    ####################################

    # test problem 1
    p = 100
    nobs = p
    nvar = p

    A = Matrix{Float64}(Matrix(I, p, p))
    b = Vector{Bool}(bitrand(p))

    progData = OptimizationMethods.LogisticRegression(A, b)
    precomp, store = OptimizationMethods.initialize(progData)
    x = zeros(Float64, p)

    ## test objective
    o = obj(progData, precomp, store, x)
    @test isapprox(o, -p*log(.5); atol = 1e-10)
    @test progData.counters.neval_obj == 1
    @test progData.counters.neval_grad == 0
    @test progData.counters.neval_hess == 0

    ## test gradient
    grad!(progData, precomp, store, x)
    g = store.grad
    @test isapprox(norm(g - ((.5) .* ones(p) - b)), 0, atol = 1e-10) 
    @test progData.counters.neval_obj == 1
    @test progData.counters.neval_grad == 1
    @test progData.counters.neval_hess == 0

    ## test objgrad
    output = objgrad!(progData, precomp, store, x)
    @test typeof(output) == Float64

    o = output
    g = store.grad
    @test isapprox(o, -p*log(.5); atol = 1e-10)
    @test isapprox(norm(g - ((.5) .* ones(p) - b)), 0, atol = 1e-10) 
    @test progData.counters.neval_obj == 2
    @test progData.counters.neval_grad == 2
    @test progData.counters.neval_hess == 0

    ## test hess
    trueHessian = .25 .* A
    OptimizationMethods.hess!(progData, precomp, store, x)
    h = store.hess
    @test isapprox(norm(h - trueHessian), 0; atol = 1e-10)
    @test progData.counters.neval_obj == 2
    @test progData.counters.neval_grad == 2
    @test progData.counters.neval_hess == 1
    
    # Test problem 2

    progData = OptimizationMethods.LogisticRegression(Float64)
    precomp, store = OptimizationMethods.initialize(progData)
    nobs, nvar = size(progData.design)
    A = progData.design
    b = progData.response
    logistic = OptimizationMethods.logistic

    function F(x::Vector)
        probs = logistic.(A * x)
        return - b' * log.(probs) - (1 .- b)' * log.(1 .- probs)
    end

    x = randn(nvar)

    ## test objective
    o = obj(progData, precomp, store, x)
    o_true = F(x)
    @test isapprox(o - o_true, 0, atol = 1e-10)

    ## test gradient
    grad!(progData, precomp, store, x)
    g = store.grad
    g_autodiff = ForwardDiff.gradient(F, x) 
    @test isapprox(norm(g - g_autodiff), 0, atol = 1e-10)

    ## test objgrad
    output = objgrad!(progData, precomp, store, x)
    o, g = output, store.grad
   
    g_autodiff = ForwardDiff.gradient(F, x)
    @test isapprox(norm(g - g_autodiff), 0, atol = 1e-10) 
    @test isapprox(norm(o - F(x)), 0, atol = 1e-10)

    ## test hess
    OptimizationMethods.hess!(progData, precomp, store, x)
    h = store.hess
    h_autodiff = ForwardDiff.hessian(F, x)
    @test isapprox(norm(h - h_autodiff), 0; atol = 1e-10)
end

end # end module
