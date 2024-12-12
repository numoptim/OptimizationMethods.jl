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
    initial_weight = 1 / 1e-2
    threshold = 1e-8

    # One Step Update 
    let progData = progData, x0 = x0, initial_weight = initial_weight,
        threshold = threshold

        optData = WeightedNormDampingGD(
            Float64, 
            x0 = x0,
            init_norm_damping_factor = initial_weight, 
            threshold = threshold,
            max_iterations = 1,
        )

        x1 = weighted_norm_damping_gd(optData, progData)

        # Test updated values 
        g0 = OptimizationMethods.grad(progData, x0)
        @test optData.stop_iteration == 1
        @test optData.grad_val_hist[1] ≈ norm(g0)
        @test optData.grad_val_hist[2] ≈ norm(OptimizationMethods.grad(progData, 
            x1))
        @test optData.iter_hist[1] ≈ x0
        @test optData.iter_hist[2] ≈ x1
        @test 1e-2 ≈ -sum((x1 - x0) ./g0) / length(x1)
    end

    let progData = progData, x0 = x0, initial_weight = initial_weight,
        threshold = threshold 

        optData = WeightedNormDampingGD(
            Float64,
            x0 = x0,
            init_norm_damping_factor = initial_weight,
            threshold = threshold,
            max_iterations = 100
        )

        xk = weighted_norm_damping_gd(optData, progData)
        
        # (Induction) Extract values assumed to be correct
        k = optData.stop_iteration + 1
        xkm1 = optData.iter_hist[k-1]
        xkm2 = optData.iter_hist[k-2]
        gkm1 = OptimizationMethods.grad(progData, xkm1)
        gkm2 = OptimizationMethods.grad(progData, xkm2)
        sskm2 = -sum((xkm1 - xkm2) ./ gkm2)/length(xkm1)
        
        # Conclusion  
        sskm1 = 1/((1/sskm2) + norm(gkm1)^2*sskm2)
        @test sskm1 ≈ -sum((xk - xkm1) ./ gkm1)/length(xkm1)
    end
end

end