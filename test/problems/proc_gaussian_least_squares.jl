# Date: 09/24/2024
# Author: Christian Varner
# Purpose: Test the implementation of GaussianLeastSquares in
# src/problems/gaussian_least_squares.jl

module ProceduralTestGaussianLeastSquares

using Test, OptimizationMethods, Random, LinearAlgebra

@testset "Problem: Gaussian Least Squares -- Procedural" begin

    # Set the seed for reproducibility
    Random.seed!(1010)

    ####################################
    # Test Struct: GaussianLeastSquares
    ####################################

    # test supertype
    @test supertype(OptimizationMethods.GaussianLeastSquares) == OptimizationMethods.AbstractNLSModel

    # test constructor -- default values
    nlp = OptimizationMethods.GaussianLeastSquares(Float64)

    ## test arguments
    @test nlp.nls_meta.nequ == 1000
    @test nlp.nls_meta.nvar == 50
    @test nlp.nls_meta.nnzj == 1000 * 50
    @test nlp.nls_meta.nnzh == 50 * 50
    @test nlp.nls_meta.lin == collect(1:1000)
    @test nlp.meta.nvar == 50

    ## test fields of structure
    @test typeof(nlp.meta.x0) == Vector{Float64}
    @test size(nlp.meta.x0) == (50, )
    @test typeof(nlp.coef) == Matrix{Float64}
    @test size(nlp.coef) == (1000, 50)
    @test typeof(nlp.cons) == Vector{Float64}
    @test size(nlp.cons)[1] == 1000

    # test constructor -- different type
    nlp = OptimizationMethods.GaussianLeastSquares(Float16)

    ## test arguments
    @test nlp.nls_meta.nequ == 1000
    @test nlp.nls_meta.nvar == 50
    @test nlp.nls_meta.nnzj == 1000 * 50
    @test nlp.nls_meta.nnzh == 50 * 50
    @test nlp.nls_meta.lin == collect(1:1000)
    
    ## test fields of structure
    @test typeof(nlp.meta.x0) == Vector{Float16}
    @test size(nlp.meta.x0) == (50, )
    @test typeof(nlp.coef) == Matrix{Float16}
    @test size(nlp.coef) == (1000, 50)
    @test typeof(nlp.cons) == Vector{Float16}
    @test size(nlp.cons)[1] == 1000

    # test constructor -- non-default values
    nequ = rand(1:1000)[1]
    nvar = rand(1:1000)[1]
    nlp = OptimizationMethods.GaussianLeastSquares(Float64; nequ = nequ, nvar = nvar)

    ## test arguments
    @test nlp.nls_meta.nequ == nequ
    @test nlp.nls_meta.nvar == nvar
    @test nlp.nls_meta.nnzj == nequ * nvar
    @test nlp.nls_meta.nnzh == nvar * nvar
    @test nlp.nls_meta.lin == collect(1:nequ)

    ## test fields of structure
    @test typeof(nlp.meta.x0) == Vector{Float64}
    @test size(nlp.meta.x0) == (nvar, )
    @test typeof(nlp.coef) == Matrix{Float64}
    @test size(nlp.coef) == (nequ, nvar)
    @test typeof(nlp.cons) == Vector{Float64}
    @test size(nlp.cons)[1] == nequ

    ####################################
    # Test Struct: PrecomputeGLS 
    ####################################

    # test supertype
    @test supertype(OptimizationMethods.PrecomputeGLS) == OptimizationMethods.AbstractPrecompute

    # test constructor -- default values
    nlp = OptimizationMethods.GaussianLeastSquares(Float16)
    precomp = OptimizationMethods.PrecomputeGLS(nlp)

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
    nlp = OptimizationMethods.GaussianLeastSquares(Float64; nequ = nequ, nvar = nvar)
    precomp = OptimizationMethods.PrecomputeGLS(nlp)

    ## test field values -- correct definitions
    @test size(precomp.coef_t_coef) == (nvar, nvar)
    @test size(precomp.coef_t_cons) == (nvar, )
    @test norm(precomp.coef_t_coef - nlp.coef' * nlp.coef) < eps()
    @test norm(precomp.coef_t_cons - nlp.coef' * nlp.cons) < eps()

    ## test field values -- correct types
    @test typeof(precomp.coef_t_coef) == Matrix{Float64}
    @test typeof(precomp.coef_t_cons) == Vector{Float64} 

    ####################################
    # Test Struct: AllocateGLS 
    ####################################

    # test supertype
    @test supertype(OptimizationMethods.AllocateGLS) == OptimizationMethods.AbstractProblemAllocate

    # test constructor -- no precomp -- default values
    nlp = OptimizationMethods.GaussianLeastSquares(Float16) 
    store = OptimizationMethods.AllocateGLS(nlp)

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
    nlp = OptimizationMethods.GaussianLeastSquares(Float32; nequ = nequ, nvar = nvar) 
    store = OptimizationMethods.AllocateGLS(nlp)

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
    nlp = OptimizationMethods.GaussianLeastSquares(Float16) 
    precomp = OptimizationMethods.PrecomputeGLS(nlp)
    store = OptimizationMethods.AllocateGLS(nlp, precomp)

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
    nlp = OptimizationMethods.GaussianLeastSquares(Float64; nequ = nequ, nvar = nvar) 
    precomp = OptimizationMethods.PrecomputeGLS(nlp)
    store = OptimizationMethods.AllocateGLS(nlp)

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
    nlp = OptimizationMethods.GaussianLeastSquares(Float16) 
    precomp = OptimizationMethods.PrecomputeGLS(nlp)
    store = OptimizationMethods.AllocateGLS(nlp) 

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

    nlp = OptimizationMethods.GaussianLeastSquares(Float64) 
    x0 = randn(50)

    # residual
    res = OptimizationMethods.residual(nlp, x0)
    @test res ≈ nlp.coef * x0 - nlp.cons
    @test nlp.counters.neval_residual == 1

    # obj
    obj = OptimizationMethods.obj(nlp, x0)
    @test obj ≈ .5 * (norm(nlp.coef * x0 - nlp.cons) ^ 2)
    @test nlp.counters.neval_obj == 1
    @test nlp.counters.neval_residual == 2

    # jac_residual
    jac = OptimizationMethods.jac_residual(nlp, x0)
    @test jac ≈ nlp.coef
    @test nlp.counters.neval_jac_residual == 1

    # grad
    grad = OptimizationMethods.grad(nlp, x0)
    @test grad ≈ nlp.coef' * nlp.coef * x0 - nlp.coef' * nlp.cons 
    @test nlp.counters.neval_jac_residual == 2
    @test nlp.counters.neval_residual == 3
    @test nlp.counters.neval_grad == 1

    # objgrad
    o, g = OptimizationMethods.objgrad(nlp, x0)
    @test obj ≈ .5 * (norm(nlp.coef * x0 - nlp.cons) ^ 2)
    @test grad ≈ nlp.coef' * nlp.coef * x0 - nlp.coef' * nlp.cons 
    @test nlp.counters.neval_obj == 2
    @test nlp.counters.neval_residual == 5
    @test nlp.counters.neval_grad == 2

    # hess
    hess = OptimizationMethods.hess(nlp, x0)
    @test hess ≈ nlp.coef' * nlp.coef
    @test nlp.counters.neval_hess == 1

    ################################################################################
    # Test Functionality: Operations - not in-place and using precomputed values
    ################################################################################

    # residual

    # obj

    # jac_residual

    # grad

    # objgrad

    # hess

    ################################################################################
    # Test Functionality: Operations - in-place and using precomputed values
    ################################################################################

    # residual!

    # obj

    # jac_residual!

    # grad!

    # objgrad!

    # hess!

end # end test

end # end module