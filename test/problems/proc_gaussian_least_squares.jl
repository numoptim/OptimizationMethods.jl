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

    ## test field types

    # test constructor -- no precomp -- non-default values

    # test constructor -- precomp -- default value

    # test constructor -- precomp -- non-default value
    
    ######################################
    # Test Functionality: initialize(...)
    ######################################


    ################################################################################
    # Test Functionality: Operations - not in-place and not using precomputed values
    ################################################################################


    ################################################################################
    # Test Functionality: Operations - not in-place and using precomputed values
    ################################################################################


    ################################################################################
    # Test Functionality: Operations - in-place and using precomputed values
    ################################################################################


end # end test

end # end module