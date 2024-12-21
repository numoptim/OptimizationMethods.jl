module TestNesterovAcceleratedGradient

using Test, OptimizationMethods, LinearAlgebra, Random 

@testset "Method: Nesterov's Accelerated Gradient" begin 

    #Testing Context 
    Random.seed!(1010)

    ##########################################
    # Test struct properties
    ##########################################

    ## test if method struct is defined
    @test @isdefined NesterovAcceleratedGD 

    ## test supertype of method struct 
    @test supertype(NesterovAcceleratedGD) == 
        OptimizationMethods.AbstractOptimizerData

    ## Test Field Names, Types and Constructors
    field_info(type::T) where T = [
        [:name, String],
        [:step_size, type],
        [:z, Vector{type}],
        [:y, Vector{type}],
        [:B, type],
        [:threshold, type],
        [:max_iterations, Int64],
        [:iter_hist, Vector{Vector{type}}],
        [:grad_val_hist, Vector{type}],
        [:stop_iteration, Int64]
    ]

    let param_dim=10, float_types=[Float16, Float32, Float64], max_iter=100
        for float_type in float_types 
            fields = field_info(float_type)
            optData = NesterovAcceleratedGD(
                float_type, 
                x0 = randn(float_type, param_dim),
                step_size = float_type(1.0),
                threshold = float_type(1e-3),
                max_iterations = max_iter
            )

            # Check Field Names and Types 
            for field_elem in fields 
                #Test Names 
                @test field_elem[1] in fieldnames(NesterovAcceleratedGD)

                #Test Types 
                @test field_elem[2] == typeof(getfield(optData, field_elem[1]))
            end

            # Test Assertions 
            ## Non-positive Step Size 
            @test_throws AssertionError NesterovAcceleratedGD(
                float_type, 
                x0 = randn(float_type, param_dim),
                step_size = float_type(-1.0),
                threshold = float_type(1e-3),
                max_iterations = max_iter
            )

            ## Non-positive threshold 
            @test_throws AssertionError NesterovAcceleratedGD(
                float_type,
                x0 = randn(float_type, param_dim),
                step_size = float_type(1.0),
                threshold = float_type(-1e-3),
                max_iterations = max_iter
            )

            ## Non-negative max iterations 
            ## Test of regular constructor will throw a bounds error rather 
            ## than an asserion error as a vector of length 0 will be constructed
            ## for the iterate history and an attempt will be made to access 
            ## its first element in the constructor in order to write x0 to 
            ## this element. 
            @test_throws AssertionError NesterovAcceleratedGD{float_type}(
                "NAGD",
                float_type(1.0), #Step Size 
                randn(float_type, param_dim), # z buffer 
                randn(float_type, param_dim), # y buffer 
                float_type(0), #B
                float_type(1e-3), #threshold 
                -1, #max_iterations 
                Vector{float_type}[Vector{float_type}(undef, param_dim) for 
                    i = 1:max_iter+1], #iterate storage 
                randn(float_type, max_iter+1), #grad storage 
                0 #stop iteration 
            )

            ## Incorrect number of Iterates stored (does not check dimension)
            @test_throws AssertionError NesterovAcceleratedGD{float_type}(
                "NAGD",
                float_type(1.0), #Step Size 
                randn(float_type, param_dim), # z buffer 
                randn(float_type, param_dim), # y buffer 
                float_type(0), #B
                float_type(1e-3), #threshold 
                max_iter, #max_iterations 
                Vector{float_type}[Vector{float_type}(undef, param_dim) for 
                    i = 1:max_iter], #iterate storage 
                randn(float_type, max_iter+1), #grad storage 
                0 #stop iteration 
            )

            ## Incorrect number of entries for gradient norm storage
            @test_throws AssertionError NesterovAcceleratedGD{float_type}(
                "NAGD",
                float_type(1.0), #Step Size 
                randn(float_type, param_dim), # z buffer 
                randn(float_type, param_dim), # y buffer 
                float_type(0), #B
                float_type(1e-3), #threshold 
                max_iter, #max_iterations 
                Vector{float_type}[Vector{float_type}(undef, param_dim) for 
                    i = 1:max_iter+1], #iterate storage 
                randn(float_type, max_iter), #grad storage 
                0 #stop iteration 
            )
            
            ## Mismatch in y and z buffer dimensions 
            @test_throws AssertionError NesterovAcceleratedGD{float_type}(
                "NAGD",
                float_type(1.0), #Step Size 
                randn(float_type, param_dim), # z buffer 
                randn(float_type, param_dim-1), # y buffer 
                float_type(0), #B
                float_type(1e-3), #threshold 
                max_iter, #max_iterations 
                Vector{float_type}[Vector{float_type}(undef, param_dim) for 
                    i = 1:max_iter+1], #iterate storage 
                randn(float_type, max_iter+1), #grad storage 
                0 #stop iteration 
            )
        end
    end

    ##########################################
    # Test optimizer
    ##########################################

    # Base Case(s)


    # Conclusion Step 
end

end