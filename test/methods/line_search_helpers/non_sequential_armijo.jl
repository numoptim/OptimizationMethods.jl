# Date: 02/14/2025
# Author: Christian Varner
# Purpose: Test the non sequential armijo funciton in
# non_sequential_armijo.jl

module TestNonsequentialArmijo

using Test, OptimizationMethods, Random

@testset "Utility -- Nonsequential Armijo Descent Check" begin

    # set seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :non_sequential_armijo_condition)

    ############################################################################
    # Results in success of descent condition
    ############################################################################

    # test 1
    reference_value = 10.0
    ρ = 1.0
    δk = 1.0
    α0k = 1.0
    norm_grad_θk = 1.0

    F_ψjk = reference_value - ρ * δk * α0k * 1.0 - 1.0 # = 8
    
    output = OptimizationMethods.non_sequential_armijo_condition(F_ψjk,
        reference_value, norm_grad_θk, ρ, δk, α0k)
    
    @test typeof(output) == Bool
    @test output == true

    # test 2
    reference_value = randn(1)[1]
    ρ = abs(randn(1)[1])
    δk = abs(randn(1)[1])
    α0k = abs(randn(1)[1])
    norm_grad_θk = abs(randn(1)[1])

    F_ψjk = reference_value - ρ * δk * α0k * (norm_grad_θk ^ 2) - abs(randn(1)[1])

    output = OptimizationMethods.non_sequential_armijo_condition(F_ψjk,
        reference_value, norm_grad_θk, ρ, δk, α0k)
    
    @test typeof(output) == Bool
    @test output == true

    ############################################################################
    # Results in failure of descent condition
    ############################################################################

    # test 1
    reference_value = 10.0
    ρ = 1.0
    δk = 1.0
    α0k = 1.0
    norm_grad_θk = 1.0

    F_ψjk = reference_value - ρ * δk * α0k * 1.0 + 1.0 # = 10 == reference_value
    
    output = OptimizationMethods.non_sequential_armijo_condition(F_ψjk,
        reference_value, norm_grad_θk, ρ, δk, α0k)
    
    @test typeof(output) == Bool
    @test output == false

    # test 2
    F_ψjk = randn(1)[1]
    ρ = abs(randn(1)[1])
    δk = abs(randn(1)[1])
    α0k = abs(randn(1)[1])
    norm_grad_θk = abs(randn(1)[1])

    reference_value = F_ψjk - ρ * δk * α0k * (norm_grad_θk ^ 2)

    output = OptimizationMethods.non_sequential_armijo_condition(F_ψjk,
        reference_value, norm_grad_θk, ρ, δk, α0k)
    
    @test typeof(output) == Bool
    @test output == false
end

end