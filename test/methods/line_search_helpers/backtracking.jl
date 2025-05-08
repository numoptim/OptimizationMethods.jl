# Date: 01/24/2025
# Author: Christian Varner
# Purpose: Test the backtracking implementation

module TestBacktracking

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Helper: Backtracking" begin

# test definitions
@test isdefined(OptimizationMethods, :backtracking!)

################################################################################
# Testing of backtracking!(...) v1 -- backtracking should be satisfied
################################################################################

let dim = 50
    Random.seed!(1010)
    progData = OptimizationMethods.LeastSquares(Float64)

    # functions required
    F(θ) = OptimizationMethods.obj(progData, θ)

    # other arguments
    θk = zeros(dim)
    θkm1 = randn(dim)
    gkm1 = OptimizationMethods.grad(progData, θkm1)
    step_direction = 100 * gkm1
    reference_value = F(θkm1)
    α = 1.0
    δ = .5
    ρ = 1e-4
    max_iteration = 100

    # backtrack
    output = OptimizationMethods.backtracking!(θk, θkm1, F, gkm1, step_direction, 
        reference_value, α, δ, ρ; max_iteration = max_iteration)

    ## test output
    @test output == true

    ## test inequality
    t = log(δ, (θk[1] - θkm1[1])/(-α * step_direction[1]) ) 
    @test F(θk) <= reference_value - ρ * (δ^t * α) * dot(gkm1, step_direction) || 
        t == max_iteration
    if t > 0
        @test F(θkm1 - (α * δ^(t-1)) .* step_direction) > reference_value - 
            ρ * (δ^(t-1) * α) * dot(gkm1, step_direction) 
    end
    @test θk ≈ θkm1 - (δ^t * α) .* step_direction
end

################################################################################
# Testing of backtracking!(...) v1 -- backtracking should not be satisfied
################################################################################

let max_iter = 5

    # Random seed for reproducibility
    Random.seed!(1010)

    # function test against
    F(θ) = θ[1]^2

    # parameters
    θk = zeros(1)
    θkm1 = Vector{Float64}([1])
    gkm1 = Vector{Float64}([2])
    step_direction = -gkm1
    reference_value = 1.0
    α = 1.0
    δ = .5
    ρ = 1.0
    
    backtracking_condition_satisfied = OptimizationMethods.backtracking!(θk, 
        θkm1, F, gkm1, step_direction, reference_value, α, δ, ρ;
        max_iteration = max_iter)

    @test typeof(backtracking_condition_satisfied) == Bool
    @test backtracking_condition_satisfied == false
    @test θk ≈ θkm1 - (δ ^ (max_iter) * α) .* step_direction
end

################################################################################
# Testing of backtracking!(...) v2 -- backtracking should be satisfied
################################################################################

let dim = 50
    Random.seed!(1010)
    progData = OptimizationMethods.LeastSquares(Float64)

    # functions required
    F(θ) = OptimizationMethods.obj(progData, θ)

    # other arguments
    θk = zeros(dim)
    θkm1 = randn(dim)
    gkm1 = OptimizationMethods.grad(progData, θkm1)
    norm_gkm1_squared = norm(gkm1)^2
    reference_value = F(θkm1)
    α = 1.0
    δ = .5
    ρ = 1e-4
    max_iteration = 100

    # backtrack
    output = OptimizationMethods.backtracking!(θk, θkm1, F, gkm1, norm_gkm1_squared, 
        reference_value, α, δ, ρ; max_iteration = max_iteration)

    ## test output
    @test output == true  

    ## test inequality
    t = log(δ, (θk[1] - θkm1[1])/(-α * gkm1[1]) ) 
    @test F(θk) <= reference_value - ρ * (δ^t * α) * norm_gkm1_squared || 
        t == max_iteration
    if t > 0
        @test F(θkm1 - (α * δ^(t-1)) .* gkm1) > reference_value - 
            ρ * (δ^(t-1) * α) * norm_gkm1_squared
    end
    @test θk ≈ θkm1 - (δ^t * α) .* gkm1
end

################################################################################
# Testing of backtracking!(...) v2 -- backtracking should not be satisfied
################################################################################

let max_iter = 5

    # Random seed for reproducibility
    Random.seed!(1010)

    # function test against
    F(θ) = θ[1]^2

    # parameters
    θk = zeros(1)
    θkm1 = Vector{Float64}([1])
    gkm1 = -Vector{Float64}([2]) ## purposely give wrong gradient so ls fails
    norm_gkm1_squared = 4.0
    reference_value = 1.0
    α = 1.0
    δ = .5
    ρ = 1.0
    
    backtracking_condition_satisfied = OptimizationMethods.backtracking!(θk, 
        θkm1, F, gkm1, norm_gkm1_squared, reference_value, α, δ, ρ;
        max_iteration = max_iter)

    @test typeof(backtracking_condition_satisfied) == Bool
    @test backtracking_condition_satisfied == false
    @test θk ≈ θkm1 - (δ ^ (max_iter) * α) .* gkm1
end

end

end