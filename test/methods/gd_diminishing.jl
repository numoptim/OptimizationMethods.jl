module TestGDDiminishingStepsize

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Method: GD Diminishing Step Size" begin
    
    #Testing Context
    Random.seed!(1010)

    ##########################################
    # Test struct properties
    ##########################################

    ## test if method struct is defined 
    @test @isdefined DiminishingStepGD
    
    ## test supertype of method struct 
    @test supertype(DiminishingStepGD) == 
        OptimizationMethods.AbstractOptimizerData

    ## Test Field Names, Types and Constructors 
    field_info(type::T) where T = [
        [:name, String],
        [:step_size_function, Function],
        [:step_size_scaling, type],
        [:threshold, type],
        [:max_iterations, Int64],
        [:iter_hist, Vector{Vector{type}}],
        [:grad_val_hist, Vector{type}],
        [:stop_iteration, Int64],
    ]

    float_types = [Float16, Float32, Float64]
    param_dim = 10 

    for float_type in float_types
        fields = field_info(float_type)
        optData = DiminishingStepGD(
            float_type,
            x0 = zeros(float_type, param_dim),
            step_size_function = OptimizationMethods.stepdown_100_step_size,
            step_size_scaling = float_type(1),
            threshold = float_type(1e-3),
            max_iterations = 100
        )

        for field_elem in fields
            #Test Names 
            @test field_elem[1] in fieldnames(DiminishingStepGD)

            #Test Types
            if field_elem[1] == :step_size_function
                @test supertype(typeof(getfield(optData, field_elem[1]))) ==
                    Function
            else
                @test field_elem[2] == typeof(getfield(optData, field_elem[1]))
            end
        end
    end

    ##########################################
    # Test optimizer
    ##########################################

    progData = OptimizationMethods.LeastSquares(Float64)
    x0 = progData.meta.x0
    ss_function = OptimizationMethods.stepdown_100_step_size
    ss_scaling = 1e-2
    threshold = 1e-3 
    
    # One Step Update 
    let progData=progData, x0=x0, step_size_function=ss_function, 
        step_size_scaling=ss_scaling, threshold=threshold

        optData = DiminishingStepGD(
            Float64, 
            x0 = x0, 
            step_size_function = step_size_function, 
            step_size_scaling = step_size_scaling, 
            threshold = threshold,
            max_iterations = 1,
        )

        x1 = diminishing_step_gd(optData, progData)

        # Test update values 
        g0 = OptimizationMethods.grad(progData, x0)
        g1 = OptimizationMethods.grad(progData, x1)
        @test optData.stop_iteration == 1
        @test optData.grad_val_hist[1] ≈ norm(g0)
        @test optData.grad_val_hist[2] ≈ norm(g1)
        @test optData.iter_hist[1] == x0 
        @test optData.iter_hist[2] == x1
        @test step_size_function(Float64, 0) * step_size_scaling ≈
            -sum((x1 - x0) ./ g0) /length(x1)
    end

    # Conclusion Step Update 
    let progData=progData, x0=x0, step_size_function=ss_function, 
        step_size_scaling=ss_scaling, threshold=threshold

        optData = DiminishingStepGD(
            Float64,
            x0 = x0,
            step_size_function = step_size_function,
            step_size_scaling = step_size_scaling,
            threshold = threshold,
            max_iterations = 100 
        )

        xk = diminishing_step_gd(optData, progData)

        # Induction Step 
        k = optData.stop_iteration 
        xkm1 = optData.iter_hist[k]
        gkm1 = OptimizationMethods.grad(progData, xkm1)
        sskm1 = step_size_function(Float64, k-1) * step_size_scaling

        # Conclusion 
        @test optData.iter_hist[k+1] == xk 
        @test sskm1 ≈ -sum((xk - xkm1) ./ gkm1) / length(xkm1)
    end




end
end