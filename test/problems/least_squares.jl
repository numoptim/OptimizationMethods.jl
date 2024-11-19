# Date: 09/24/2024
# Author: Christian Varner
# Purpose: Test the implementation of LeastSquares in
# src/problems/least_squares.jl

module TestLeastSquares

using Test, OptimizationMethods, Random, LinearAlgebra, NLPModels

@testset "Problem: Least Squares" begin

    # Set the seed for reproducibility
    Random.seed!(1010)

    ####################################
    # Test Struct: Least Squares
    ####################################

    # check if struct is defined 
    @test isdefined(OptimizationMethods, :LeastSquares)

    # test supertype
    @test supertype(OptimizationMethods.LeastSquares) == 
        AbstractNLSModel

    # test fields
    field_names = [:meta, :nls_meta, :counters, :coef, :cons]

    for name in field_names
        @test name in fieldnames(OptimizationMethods.LeastSquares)
    end 

    ####################################
    # Test constructor 
    ####################################
    real_types = [Float16, Float32, Float64]
    nobs = 1000
    nvar = 50 


    # Test Generative Constructor 
    for real_type in real_types
        progData = OptimizationMethods.LeastSquares(real_type)

        # Test Type and Value: coef 
        @test typeof(progData.coef) == Matrix{real_type}
        @test size(progData.coef) == (nobs, nvar)

        # Test Type and Value: const
        @test typeof(progData.cons) == Vector{real_type}
        @test length(progData.cons) == nobs

        # Test Type and Value: x0
        @test typeof(progData.meta.x0) == Vector{real_type}
        @test length(progData.meta.x0) == nvar
    end
    
    # Test Data-supplied Constructor 
    real_types = [Float16, Float32, Float64]
    nobs = 100
    nvar = 5
    nobs_error = 90
    nvar_error = 3

    for real_type in real_types

        ## test that error gets thrown -- incorrect response dimension
        A = rand(real_type, nobs, nvar)
        b = rand(real_type, nobs_error)
        
        @test_throws AssertionError OptimizationMethods.LeastSquares(A, b)  

        ## test that error gets thrown -- incorrect initial point 
        ## dimension
        b = randn(real_type, nobs)
        x = randn(real_type, nvar_error)
        
        @test_throws NLPModels.DimensionError OptimizationMethods.LeastSquares(
            A, b; x0 = x) 
            "Input x0 should have length $(nvar) not $(nvar_error)"

        ## test the types and correct values
        x = randn(real_type, nvar)        
        progData = OptimizationMethods.LeastSquares(A, b; x0 = x)

        @test typeof(progData.coef) == Matrix{real_type}
        @test typeof(progData.cons) == Vector{real_type}

        ### field values
        @test progData.coef == A
        @test progData.cons == b

        ### meta is correctly initialized
        @test progData.meta.x0 == x
        @test progData.meta.name == "Least Squares"
        @test progData.meta.nvar == nvar
        @test progData.nls_meta.nequ == nobs
        @test progData.nls_meta.nvar == nvar
        @test progData.nls_meta.nnzj == nobs * nvar
        @test progData.nls_meta.nnzh == nvar * nvar
        @test progData.nls_meta.lin == collect(1:nobs)
    end

    ####################################
    # Test Struct: PrecomputeLS 
    ####################################

    # test supertype
    @test supertype(OptimizationMethods.PrecomputeLS) == 
        OptimizationMethods.AbstractPrecompute

    # test fields
    @test :coef_t_coef in fieldnames(OptimizationMethods.PrecomputeLS)
    @test :coef_t_cons in fieldnames(OptimizationMethods.PrecomputeLS)

    # test constructor -- default values
    nlp = OptimizationMethods.LeastSquares(Float16)
    precomp = OptimizationMethods.PrecomputeLS(nlp)

    ## test field values -- correct definitions
    @test size(precomp.coef_t_coef) == (nlp.nls_meta.nvar, nlp.nls_meta.nvar)
    @test size(precomp.coef_t_cons) == (nlp.nls_meta.nvar, )
    @test norm(precomp.coef_t_coef - nlp.coef' * nlp.coef) < eps()
    @test norm(precomp.coef_t_cons - nlp.coef' * nlp.cons) < eps()

    ## test field values -- correct types
    @test typeof(precomp.coef_t_coef) == Matrix{Float16}
    @test typeof(precomp.coef_t_cons) == Vector{Float16} 

    # test constructor -- non-default values
    nequ = rand(1:1000)[1]
    nvar = rand(1:1000)[1]
    nlp = OptimizationMethods.LeastSquares(Float64; nequ = nequ, nvar = nvar)
    precomp = OptimizationMethods.PrecomputeLS(nlp)

    ## test field values -- correct definitions
    @test size(precomp.coef_t_coef) == (nvar, nvar)
    @test size(precomp.coef_t_cons) == (nvar, )
    @test norm(precomp.coef_t_coef - nlp.coef' * nlp.coef) < eps()
    @test norm(precomp.coef_t_cons - nlp.coef' * nlp.cons) < eps()

    ## test field values -- correct types
    @test typeof(precomp.coef_t_coef) == Matrix{Float64}
    @test typeof(precomp.coef_t_cons) == Vector{Float64} 

    ####################################
    # Test Struct: AllocateLS 
    ####################################

    # test supertype
    @test supertype(OptimizationMethods.AllocateLS) == 
        OptimizationMethods.AbstractProblemAllocate

    # test fields
    @test :res in fieldnames(OptimizationMethods.AllocateLS)
    @test :jac in fieldnames(OptimizationMethods.AllocateLS)
    @test :grad in fieldnames(OptimizationMethods.AllocateLS)
    @test :hess in fieldnames(OptimizationMethods.AllocateLS)

    # test constructor -- no precomp -- default values
    nlp = OptimizationMethods.LeastSquares(Float16) 
    store = OptimizationMethods.AllocateLS(nlp)

    ## test field
    @test store.res == zeros(Float16, nlp.nls_meta.nequ)
    @test store.jac == nlp.coef
    @test store.grad == zeros(Float16, nlp.meta.nvar)
    @test store.hess == nlp.coef' * nlp.coef

    ## test field types
    @test typeof(store.res) == Vector{Float16}
    @test typeof(store.jac) == Matrix{Float16}
    @test typeof(store.grad) == Vector{Float16}
    @test typeof(store.hess) == Matrix{Float16}

    # test constructor -- no precomp -- non-default values
    nequ = rand(1:1000)[1]
    nvar = rand(1:1000)[1] 
    nlp = OptimizationMethods.LeastSquares(Float32; nequ = nequ, nvar = nvar) 
    store = OptimizationMethods.AllocateLS(nlp)

    ## test field
    @test store.res == zeros(Float32, nlp.nls_meta.nequ)
    @test store.jac == nlp.coef
    @test store.grad == zeros(Float32, nlp.meta.nvar)
    @test store.hess == nlp.coef' * nlp.coef

    ## test field types
    @test typeof(store.res) == Vector{Float32}
    @test typeof(store.jac) == Matrix{Float32}
    @test typeof(store.grad) == Vector{Float32}
    @test typeof(store.hess) == Matrix{Float32}

    # test constructor -- precomp -- default value
    nlp = OptimizationMethods.LeastSquares(Float16) 
    precomp = OptimizationMethods.PrecomputeLS(nlp)
    store = OptimizationMethods.AllocateLS(nlp, precomp)

    ## test field
    @test store.res == zeros(Float16, nlp.nls_meta.nequ)
    @test store.jac == nlp.coef
    @test store.grad == zeros(Float16, nlp.meta.nvar)
    @test store.hess == precomp.coef_t_coef

    ## test field types
    @test typeof(store.res) == Vector{Float16}
    @test typeof(store.jac) == Matrix{Float16}
    @test typeof(store.grad) == Vector{Float16}
    @test typeof(store.hess) == Matrix{Float16}

    # test constructor -- precomp -- non-default value
    nequ = rand(1:1000)[1]
    nvar = rand(1:1000)[1] 
    nlp = OptimizationMethods.LeastSquares(Float64; nequ = nequ, nvar = nvar) 
    precomp = OptimizationMethods.PrecomputeLS(nlp)
    store = OptimizationMethods.AllocateLS(nlp)

    ## test field
    @test store.res == zeros(Float64, nlp.nls_meta.nequ)
    @test store.jac == nlp.coef
    @test store.grad == zeros(Float64, nlp.meta.nvar)
    @test store.hess == precomp.coef_t_coef

    ## test field types
    @test typeof(store.res) == Vector{Float64}
    @test typeof(store.jac) == Matrix{Float64}
    @test typeof(store.grad) == Vector{Float64}
    @test typeof(store.hess) == Matrix{Float64}
    
    ######################################
    # Test Functionality: initialize(...)
    ######################################

    # testing with default values
    nlp = OptimizationMethods.LeastSquares(Float16) 
    precomp = OptimizationMethods.PrecomputeLS(nlp)
    store = OptimizationMethods.AllocateLS(nlp) 

    ## initialize storage and precomputed values
    init_precompute, init_store = OptimizationMethods.initialize(nlp)

    ## check fields of precompute
    @test init_precompute.coef_t_coef == precomp.coef_t_coef
    @test init_precompute.coef_t_cons == precomp.coef_t_cons

    ## check types of store
    @test typeof(init_precompute.coef_t_coef) == Matrix{Float16}
    @test typeof(init_precompute.coef_t_cons) == Vector{Float16}

    ## check fields of store
    @test init_store.res == store.res
    @test init_store.jac == store.jac
    @test init_store.grad == store.grad
    @test init_store.hess == store.hess

    ## check types of store
    @test typeof(init_store.res) == Vector{Float16}
    @test typeof(init_store.jac) == Matrix{Float16}
    @test typeof(init_store.grad) == Vector{Float16}
    @test typeof(init_store.hess) == Matrix{Float16}

    ################################################################################
    # Test Functionality: Operations - not in-place and not using precomputed values
    ################################################################################

    nlp = OptimizationMethods.LeastSquares(Float32) 
    x0 = randn(Float32, 50)

    # residual
    res = OptimizationMethods.residual(nlp, x0)
    @test res ≈ nlp.coef * x0 - nlp.cons
    @test nlp.counters.neval_residual == 1
    @test typeof(res) == Vector{Float32}

    # obj
    obj = OptimizationMethods.obj(nlp, x0)
    @test obj ≈ .5 * (norm(nlp.coef * x0 - nlp.cons) ^ 2)
    @test nlp.counters.neval_obj == 1
    @test nlp.counters.neval_residual == 2
    @test typeof(obj) == Float32

    # jac_residual
    jac = OptimizationMethods.jac_residual(nlp, x0)
    @test jac ≈ nlp.coef
    @test nlp.counters.neval_jac_residual == 1
    @test typeof(jac) == Matrix{Float32}

    # grad
    grad = OptimizationMethods.grad(nlp, x0)
    @test grad ≈ nlp.coef' * nlp.coef * x0 - nlp.coef' * nlp.cons 
    @test nlp.counters.neval_jac_residual == 2
    @test nlp.counters.neval_residual == 3
    @test nlp.counters.neval_grad == 1
    @test typeof(grad) == Vector{Float32}

    # objgrad
    o, g = OptimizationMethods.objgrad(nlp, x0)
    @test o ≈ .5 * (norm(nlp.coef * x0 - nlp.cons) ^ 2)
    @test grad ≈ nlp.coef' * nlp.coef * x0 - nlp.coef' * nlp.cons 
    @test nlp.counters.neval_obj == 2
    @test nlp.counters.neval_residual == 5
    @test nlp.counters.neval_grad == 2
    @test typeof(o) == Float32
    @test typeof(g) == Vector{Float32}

    # hess
    hess = OptimizationMethods.hess(nlp, x0)
    @test hess ≈ nlp.coef' * nlp.coef
    @test nlp.counters.neval_hess == 1
    @test typeof(hess) == Matrix{Float32}

    ################################################################################
    # Test Functionality: Operations - not in-place and using precomputed values
    ################################################################################

    nlp = OptimizationMethods.LeastSquares(Float16)
    precomp, store = OptimizationMethods.initialize(nlp)
    x0 = randn(Float16, 50)

    # residual
    res = nlp.coef * x0 - nlp.cons
    returned_res = OptimizationMethods.residual(nlp, precomp, x0)
    @test res ≈ returned_res
    @test nlp.counters.neval_residual == 1
    @test typeof(returned_res) == Vector{Float16}

    # obj
    obj = .5 * dot(res, res)
    returned_obj = OptimizationMethods.obj(nlp, precomp, x0)
    @test obj ≈ returned_obj
    @test nlp.counters.neval_residual == 2
    @test nlp.counters.neval_obj == 1
    @test typeof(returned_obj) == Float16

    # jac_residual
    jac = nlp.coef
    returned_jac = OptimizationMethods.jac_residual(nlp, precomp, x0)
    @test jac == returned_jac
    @test nlp.counters.neval_jac_residual == 1
    @test typeof(returned_jac) == Matrix{Float16}

    # grad
    grad = nlp.coef' * nlp.coef * x0 - nlp.coef' * nlp.cons
    returned_grad = OptimizationMethods.grad(nlp, precomp, x0)
    @test grad ≈ returned_grad
    @test nlp.counters.neval_grad == 1
    @test typeof(returned_grad) == Vector{Float16}

    # objgrad
    returned_obj, returned_grad = OptimizationMethods.objgrad(nlp, precomp, x0)
    @test obj ≈ returned_obj 
    @test grad ≈ returned_grad
    @test nlp.counters.neval_residual == 3
    @test nlp.counters.neval_obj == 2
    @test nlp.counters.neval_grad == 2
    @test typeof(returned_obj) == Float16
    @test typeof(returned_grad) == Vector{Float16}

    # hess
    hess = nlp.coef' * nlp.coef
    returned_hess = OptimizationMethods.hess(nlp, precomp, x0)
    @test hess ≈ returned_hess
    @test nlp.counters.neval_hess == 1
    @test typeof(hess) == Matrix{Float16}

    ################################################################################
    # Test Functionality: Operations - in-place and using precomputed values
    ################################################################################

    nlp = OptimizationMethods.LeastSquares(Float32)
    precomp, store = OptimizationMethods.initialize(nlp)
    x0 = randn(Float32, 50)

    # residual!
    res = nlp.coef * x0 - nlp.cons
    val = OptimizationMethods.residual!(nlp, precomp, store, x0)
    @test isnothing(val)
    @test res ≈ store.res
    @test nlp.counters.neval_residual == 1
    @test typeof(store.res) == Vector{Float32} 

    # obj - recompute True
    obj = .5 * res' * res
    returned_obj = OptimizationMethods.obj(nlp, precomp, store, x0)
    @test obj ≈ returned_obj
    @test nlp.counters.neval_residual == 2
    @test nlp.counters.neval_obj == 1
    @test typeof(returned_obj) == Float32

    # obj - recompute False
    returned_obj = OptimizationMethods.obj(nlp, precomp, store, x0; recompute = false)
    @test obj ≈ returned_obj
    @test nlp.counters.neval_residual == 2
    @test nlp.counters.neval_obj == 2
    @test typeof(returned_obj) == Float32

    # jac_residual!

    ## jac_residual! - recompute true
    jac = nlp.coef
    val = OptimizationMethods.jac_residual!(nlp, precomp, store, x0)
    @test isnothing(val)
    @test jac == store.jac
    @test nlp.counters.neval_jac_residual == 1
    @test typeof(jac) == Matrix{Float32}

    ## jac_residual! -- recompute false
    jac = nlp.coef
    val = OptimizationMethods.jac_residual!(nlp, precomp, store, x0; recompute = false)
    @test isnothing(val)
    @test jac == store.jac
    @test nlp.counters.neval_jac_residual == 2
    @test typeof(jac) == Matrix{Float32} 

    # grad!

    ## grad! -- recompute true
    grad = nlp.coef' * nlp.coef * x0 - nlp.coef' * nlp.cons
    val = OptimizationMethods.grad!(nlp, precomp, store, x0)
    @test isnothing(val)
    @test grad ≈ store.grad
    @test nlp.counters.neval_grad == 1
    @test typeof(grad) == Vector{Float32}

    ## grad! -- recompute false
    val = OptimizationMethods.grad!(nlp, precomp, store, x0; recompute = false)
    @test isnothing(val)
    @test grad ≈ store.grad
    @test nlp.counters.neval_grad == 2
    @test typeof(grad) == Vector{Float32}

    # objgrad!

    ## recompute true
    o = OptimizationMethods.objgrad!(nlp, precomp, store, x0)
    @test abs(obj - o) < 1e-2
    @test grad ≈ store.grad
    @test nlp.counters.neval_residual == 2
    @test nlp.counters.neval_obj == 3
    @test nlp.counters.neval_grad == 3
    @test typeof(o) == Float32
    @test typeof(store.grad) == Vector{Float32}

    # hess!

    ## recompute true
    hess = nlp.coef' * nlp.coef
    val = OptimizationMethods.hess!(nlp, precomp, store, x0)
    @test hess ≈ store.hess
    @test nlp.counters.neval_hess == 1
    @test typeof(hess) == Matrix{Float32}

    ## recompute false
    hess = nlp.coef' * nlp.coef
    val = OptimizationMethods.hess!(nlp, precomp, store, x0; recompute = false)
    @test hess ≈ store.hess
    @test nlp.counters.neval_hess == 2
    @test typeof(hess) == Matrix{Float32}

end # end test

end # end module