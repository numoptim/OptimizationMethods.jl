# Date: 01/28/2025
# Author: Christian Varner
# Purpose: Test the non-monotone line search utility in 
# nls_cache_util.jl

module TestNLSCacheUtil

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Utility: NLS Cache Functions" begin

################################################################################
# Test cases for shift_left!(...)
################################################################################

# test definition
@test isdefined(OptimizationMethods, :shift_left!)

# test function
dim = [1, 10, 100]
let dimensions = dim
    for dim in dimensions

        # generate random vector and shift
        array = randn(dim)
        copy_array = copy(array)
        OptimizationMethods.shift_left!(array, dim)

        # output should be the same as circ
        @test circshift(copy_array, -1) == array
    end
end

################################################################################
# Test cases for update_maximum(...)
################################################################################

# test definition
@test isdefined(OptimizationMethods, :update_maximum)

dim = [1, 10, 100]
let dimensions = dim
    for dim in dimensions

        # generate data for test case
        array = randn(dim)
        new_value = randn(1)[1]
        max_value, max_index = findmax(array)

        # shift and add value
        OptimizationMethods.shift_left!(array, dim)
        array[dim] = new_value

        # find the new maximum
        output = OptimizationMethods.update_maximum(array, max_index - 1, dim)

        # test cases
        @test length(output) == 2
        @test sum(array .<= output[1]) == dim
        @test sum(array .<= array[output[2]]) == dim
        @test (output[1] in array)
        @test (output[1] == array[output[2]])
    end
end

end

end 