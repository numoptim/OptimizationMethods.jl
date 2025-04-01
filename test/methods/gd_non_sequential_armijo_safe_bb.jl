# Date: 2025/04/01
# Author: Christian Varner
# Purpose: Test the implementation of non-sequential armijo
# gradient descent with barzilai-borwein step sizes

module TestNonsequentialArmijoBBGD

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Method -- GD with Nonsequential Armijo and BB Steps: struct" begin

    # set seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :NonsequentialArmijoSafeBBGD)
    
    # test field values -- default names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist,
        :grad_val_hist, :stop_iteration]
    let fields = default_fields
        for field_name in fields
            @test field_name in fieldnames(NonsequentialArmijoFixedGD)
        end
    end

    # test field values -- unique names

    # test field types

    # test struct initializations

    # test struct error
end

@testset "Utility -- Update Algorithm Parameters (GDNonseqArmijoBB)" begin

    include("../utility/update_algorithm_parameters_test_cases.jl")

    # arguments
    dim = 50
    x0 = randn(dim)
    long_stepsize = true
    α_lower = abs(randn())
    α_upper = α_lower + 1
    init_stepsize = α_lower + .5
    δ0 = abs(randn())
    δ_upper = δ0 + 1
    ρ = abs(randn())
    M = rand(1:100)
    threshold = abs(randn())
    max_iterations = rand(1:100)

    # build structure
    optData = NonsequentialArmijoSafeBBGD(Float64;
        x0 = x0, init_stepsize = init_stepsize, long_stepsize = long_stepsize,
        α_lower = α_lower, α_upper = α_upper, δ0 = δ0, δ_upper = δ_upper,
        ρ = ρ, M = M, threshold = threshold, max_iterations = max_iterations)

    # Conduct test cases
    update_algorithm_parameters_test_cases(optData, dim, max_iterations)
end

@testset "Utility -- Inner Loop (GDNonseqArmijoBB)" begin
end

@testset "Method -- GD with Nonsequential Armijo and BB Steps: method" begin
end

end # End module