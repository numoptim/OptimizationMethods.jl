# Date: 2025/04/16
# Author: Christian Varner
# Purpose: Test the two functions in the modified_newton helper
# file

module TestModifiedNewtonHelpers

using Test, OptimizationMethods, LinearAlgebra

@testset "Test add_identity" begin

    # generate a random matrix
    dim = 50
    A = randn(dim, dim)

    # generate a random constant to add to diagonal
    λ = randn()

    # test case
    let dim = dim, A = A, λ = λ 
        
        # apply the function
        A_copy = copy(A)
        res = OptimizationMethods.add_identity!(A, λ)

        # check matrix updated and output of function
        @test isnothing(res)
        for i in 1:dim
            for j in 1:dim
                if i == j 
                    @test A_copy[i, i] + λ == A[i, i]
                else
                    @test A_copy[i, j] == A[i, j]
                end
            end
        end
    end
    
end # end test for add_identity

@testset "Test add_identity_until_pd" begin

    # Check iteration 0
    dim = 50
    A = Matrix{Float64}(Hermitian(randn(dim, dim)))
    λ = rand()
    β = rand()
    max_iteration = 0

    let dim = dim, A = A, λ = λ, β = β,
        max_iteration = max_iteration
        
        # run iteration 0 of the method
        A_copy = copy(A)
        res = OptimizationMethods.add_identity_until_pd!(A; λ = λ, β = β, 
            max_iterations = max_iteration)

        # test for correctness 
        @test A == A_copy + (λ .* Matrix{Float64}(I, dim, dim))
        @test res[1] == λ
        @test res[2] == false
    end

    # Correct output for a successful termination
    dim = 50
    L = zeros(dim, dim)
    for i in 1:dim
        for j in 1:i
            L[i,j] = randn()
        end
    end
    A = L * L' 
    
    ## should terminate after one iteration
    let dim = dim, A = A, λ = 0.0, β = rand(),
        max_iteration = 100

        # attempt to find modification
        A_copy = copy(A)
        res = OptimizationMethods.add_identity_until_pd!(A;
            λ = λ, β = β, max_iterations = max_iteration)

        # test output
        @test res[1] == λ
        @test res[2] == true

        # get outputted cholesky factor
        L_return = zeros(dim, dim)
        for i in 1:dim
            for j in i:dim
                L_return[i,j] = A[i, j]
            end
        end

        # check the returned cholesky 
        @test L_return' * L_return ≈ 
            A_copy + (β .* Matrix{Float64}(I, dim, dim))
    end

    ## Example will terminate after a couple iterations
    dim = 50
    L = zeros(dim, dim)
    for i in 1:dim
        for j in 1:i
            L[i,j] = randn()
        end
    end
    A = L * L' - 1000 * Matrix{Float64}(I, dim, dim) 
    
    let dim = dim, A = A, λ = 0.0, β = rand(),
        max_iteration = 100

        # attempt to find modification
        A_copy = copy(A)
        res = OptimizationMethods.add_identity_until_pd!(A;
            λ = λ, β = β, max_iterations = max_iteration)

        # test output
        @test res[2] == true

        # get the returned cholesky factor
        L_return = zeros(dim, dim)
        for i in 1:dim
            for j in i:dim
                L_return[i,j] = A[i, j]
            end
        end

        # test the output of our res and the retunred cholesky 
        @test L_return' * L_return ≈ 
            A_copy + ((res[1] + β) .* Matrix{Float64}(I, dim, dim))

        # test smallest such constant to satisfy our condition
        B = A_copy + ((res[1]/10 + β) .* Matrix{Float64}(I, dim, dim))
        @test !issuccess(cholesky(Hermitian(B); check = false)) 
    end
    
    # Correct output for an unsuccessful termination 
    dim = 50
    A = -1e11 * Matrix{Float64}(I, dim, dim)
    λ = 0.0
    β = rand()
    max_iteration = rand(1:10)

    let dim = dim, A = A, λ = λ, β = β, max_iteration = max_iteration
        
        # apply method
        A_copy = copy(A)
        res = OptimizationMethods.add_identity_until_pd!(A;
            λ = λ, β = β, max_iterations = max_iteration)
        
        # check to make sure method fails
        @test res[2] == false
        @test res[1] ≈ β * (10^(max_iteration - 1))
    end
    
end # end test for add_identity_until_pd

end # end of module