# Date: 10/05/2024
# Author: Christian Varner
# Purpose: Implement some test cases for logistic regression

module TestPoissonRegression

using Test, OptimizationMethods, Random, LinearAlgebra

@testset "Problem: Logistic Regression" begin

    # set the seed for reproducibility
    Random.seed!(1010)

    ####################################
    # Test Struct: Logistic Regression
    ####################################

    # test super type
    @test supertype(OptimizationMethods.LogisticRegression) == AbstractNLPModel

    # test fields
    field_names = [:]

    ####################################
    # Test Struct End
    ####################################


end

end # end module