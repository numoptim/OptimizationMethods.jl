# OptimizationMethods.jl 

module TestWNGrad

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Method: Weighted Norm Damping GD (WNGrad)" begin

    #Testing Context 
    Random.seed!(1010)

    ##########################################
    # Test struct properties
    ##########################################

    ## test if method struct is defined 
    @test @isdefined WeightedNormDampingGD

    ## test supertype of method struct 
    @test supertype(WeightedNormDampingGD) == 
        OptimizationMethods.AbstractOptimizerData

    ## Test Field Names, Types and Constructors
    field_info(type::T) where T = [
        [:name, String],
        [:init_norm_damping_factor, type],
        [:threshold, type],
        [:max_iterations, Int64],
        [:iter_hist, Vector{Vector{type}}],
        [:grad_val_hist, Vector{type}],
        [:stop_iteration, Int64]
    ]

    let param_dim = 10, float_types = [Float16, Float32, Float64], max_iter = 100
        for float_type in float_types 
            fields = field_info(float_type)
            optData = WeightedNormDampingGD(
                float_type, 
                x0 = zeros(float_type, param_dim),
                init_norm_damping_factor = float_type(1.0),
                threshold = float_type(1e-4),
                max_iterations = max_iter
            )

            for field_elem in fields
                #Test names 
                @test field_elem[1] in fieldnames(WeightedNormDampingGD)

                #Test Types 
                @test field_elem[2] == typeof(getfield(optData, field_elem[1]))
            end

            # Test Assertions
            ## Initial norm damping factor is non-positive 
            @test_throws AssertionError optData = WeightedNormDampingGD(
                float_type, 
                x0 = zeros(float_type, param_dim),
                init_norm_damping_factor = float_type(0.0),
                threshold = float_type(1e-4),
                max_iterations = max_iter
            )

            ## Threshold is non-positive 
            @test_throws AssertionError optData = WeightedNormDampingGD(
                float_type, 
                x0 = zeros(float_type, param_dim),
                init_norm_damping_factor = float_type(1.0),
                threshold = float_type(0.0),
                max_iterations = max_iter
            )

            ## Max Iterations is non-positive
            @test_throws AssertionError optData = WeightedNormDampingGD(
                float_type, 
                x0 = zeros(float_type, param_dim),
                init_norm_damping_factor = float_type(1.0),
                threshold = float_type(1e-4),
                max_iterations = 0
            )
        end
    end

    ##########################################
    # Test optimizer
    ##########################################

    progData = OptimizationMethods.LeastSquares(Float64)
    x0 = progData.meta.x0
    inital_weight = 1 / 1e-2

    #TODO: One Step Update 
    let 

    end

    #TODO: Iteration K Update 
    let

    end
end

end