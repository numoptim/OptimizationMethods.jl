# Date: 2025/04/16
# Author: Christian Varner
# Purpose: Test upper and lower triangle solve functions

module TestTriangleSolve

using Test, OptimizationMethods, LinearAlgebra

@testset "Test upper_triangle_solve!" begin

    # test the function is defined
    @test isdefined(OptimizationMethods, :upper_triangle_solve!)

    # generate an upper triangular matrix with random normal entries
    dim = 50
    A = zeros(dim, dim)
    for i in 1:dim
        for j in i:dim
            A[i, j] = randn()
        end
    end
    
    # generate a constant vector to create a linear system
    x = randn(dim)
    b = A * x

    # test the output of upper_triangle_solve!
    let A = A, b = b

        # copy the constant vector as function overwrites
        b_copy = copy(b)
        res = OptimizationMethods.upper_triangle_solve!(b_copy, A)

        # test the output of the method
        @test isnothing(res)
        @test A * b_copy ≈ b
    end
    
end # End testing for upper_triangle_solve!

@testset "Test lower_triangle_solve!" begin

    # test the function is defined
    @test isdefined(OptimizationMethods, :lower_triangle_solve!)

    # generate a lower triangular matrix with random normal entries
    dim = 50
    A = zeros(dim, dim)
    for i in 1:dim
        for j in 1:i
            A[i, j] = randn()
        end
    end

    # generate a constant vector to create a linear system
    x = randn(dim)
    b = A * x

    # test the output of lower_triangle_solve!
    let A = A, b = b
        
        # copy the constant vector as the function overwrites values
        b_copy = copy(b)
        res = OptimizationMethods.lower_triangle_solve!(b_copy, A)

        # test the output of the method
        @test isnothing(res)
        @test A * b_copy ≈ b
    end
    
end # End testing for lower_triangle_solve!

end # End TestTriangleSolve