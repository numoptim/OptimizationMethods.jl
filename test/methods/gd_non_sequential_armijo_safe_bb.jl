# Date: 2025/04/01
# Author: Christian Varner
# Purpose: Test the implementation of non-sequential armijo
# gradient descent with barzilai-borwein step sizes

module TestNonsequentialArmijoBBGD

using Test, OptimizationMethods, CircularArrays, LinearAlgebra, Random

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
    unique_fields = [:∇F_θk, :norm_∇F_ψ, :init_stepsize, :bb_step_size,
        :α0k, :α_lower, :α_default, :iter_diff_checkpoint, :grad_diff_checkpoint,
        :iter_diff, :grad_diff, :δk, :δ_upper, :ρ, :objective_hist,
        :reference_value, :reference_value_index, :acceptance_cnt,
        :τ_lower, :τ_upper, :second_acceptance_occurred]
    let field = unique_fields
        for field_name in field
            @test field_name in fieldnames(NonsequentialArmijoSafeBBGD)
        end
    end

    # test field types
    field_info(type::T) where T = [
        [:name, String],
        [:∇F_θk, Vector{type}],
        [:norm_∇F_ψ, type],
        [:init_stepsize, type],
        [:bb_step_size, Any],
        [:α0k, type],
        [:α_lower, type],
        [:α_default, type],
        [:iter_diff_checkpoint, Vector{type}],
        [:grad_diff_checkpoint, Vector{type}],
        [:iter_diff, Vector{type}],
        [:grad_diff, Vector{type}],
        [:δk, type],
        [:δ_upper, type],
        [:ρ, type],
        [:objective_hist, CircularVector{type, Vector{type}}],
        [:reference_value, type],
        [:reference_value_index, Int64],
        [:acceptance_cnt, Int64],
        [:τ_lower, type],
        [:τ_upper, type],
        [:second_acceptance_occurred, Bool],
        [:threshold, type],
        [:max_iterations, Int64],
        [:iter_hist, Vector{Vector{type}}],
        [:grad_val_hist, Vector{type}],
        [:stop_iteration, Int64],
    ]
    real_types = [Float16, Float32, Float64]
    let field_info = field_info,
        real_types = real_types

        for type in real_types
            
            # random field values -- long_stepsize = true
            x0 = randn(type, 10)
            long_stepsize = true
            α_lower = abs(rand(type))
            α_default = type(1.0)
            init_stepsize = type((α_lower + 1/α_lower)/2)
            δ0 = abs(rand(type))
            δ_upper = δ0 + 1
            ρ = abs(rand(type))
            M = rand(1:100)
            threshold = abs(rand(type))
            max_iterations =  rand(1:100)

            optData = NonsequentialArmijoSafeBBGD(type; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = max_iterations
                )

            for (field_key, field_type) in field_info(type)
                if field_key != :bb_step_size
                    @test field_type == typeof(getfield(optData, field_key))
                end
            end
            
            # random field values -- long_stepsize = false
            x0 = randn(type, 10)
            long_stepsize = false
            α_lower = abs(rand(type))
            α_default = type(1.0)
            init_stepsize = type((α_lower + 1/α_lower)/2)
            δ0 = abs(rand(type))
            δ_upper = δ0 + 1
            ρ = abs(rand(type))
            M = rand(1:100)
            threshold = abs(rand(type))
            max_iterations =  rand(1:100)
    
            optData = NonsequentialArmijoSafeBBGD(type; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = max_iterations
                )
            
            for (field_key, field_type) in field_info(type)
                if field_key != :bb_step_size
                    @test field_type == typeof(getfield(optData, field_key))
                end
            end
        end
    end

    # test struct initializations -- long_stepsize = true
    x0 = randn(10)
    long_stepsize = true
    α_lower = abs(rand())
    α_default = 1.0
    init_stepsize = (α_lower + 1/α_lower)/2
    δ0 = abs(rand())
    δ_upper = δ0 + 1
    ρ = rand()
    M = rand(1:100)
    threshold = abs(rand())
    max_iterations = rand(1:100) 

    let x0 = x0, long_stepsize = long_stepsize, α_lower = α_lower, 
        α_default = α_default, init_stepsize = init_stepsize, 
        δ0 = δ0, δ_upper = δ_upper, ρ = ρ, M = M, threshold = threshold,
        max_iterations = max_iterations

        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations
            )

        # test field value correct initialized long_stepsize = true
        @test optData.iter_hist[1] == x0
        @test optData.init_stepsize == init_stepsize
        @test optData.bb_step_size == OptimizationMethods.bb_long_step_size
        @test optData.α_lower == α_lower
        @test optData.α_default == α_default
        @test optData.δk == δ0
        @test optData.δ_upper == δ_upper
        @test optData.ρ == ρ
        @test length(optData.objective_hist) == M
        @test optData.threshold == threshold
        @test optData.max_iterations == max_iterations
    end
    
    # random field values -- long_stepsize = false
    x0 = randn(10)
    long_stepsize = false
    α_lower = abs(rand())
    α_default = 1.0
    init_stepsize = (α_lower + 1/α_lower)/2
    δ0 = abs(rand())
    δ_upper = δ0 + 1
    ρ = rand()
    M = rand(1:100)
    threshold = abs(rand())
    max_iterations = rand(1:100)
    
    let x0 = x0, long_stepsize = long_stepsize, α_lower = α_lower, 
        α_default = α_default, init_stepsize = init_stepsize, 
        δ0 = δ0, δ_upper = δ_upper, ρ = ρ, M = M, threshold = threshold,
        max_iterations = max_iterations

        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations
            )

        # test field value correctly initialized long_stepsize = false
        @test optData.iter_hist[1] == x0
        @test optData.init_stepsize == init_stepsize
        @test optData.bb_step_size == OptimizationMethods.bb_short_step_size
        @test optData.α_lower == α_lower
        @test optData.α_default == α_default
        @test optData.δk == δ0
        @test optData.δ_upper == δ_upper
        @test optData.ρ == ρ
        @test length(optData.objective_hist) == M
        @test optData.threshold == threshold
        @test optData.max_iterations == max_iterations
        @test optData.second_acceptance_occurred == false
    end

    # test struct error
    x0 = randn(10)
    long_stepsize = true
    α_lower = abs(rand())
    α_default = 1.0
    init_stepsize = (α_lower + 1/α_lower)/2
    δ0 = abs(rand())
    δ_upper = δ0 + 1
    ρ = rand()
    M = rand(1:100)
    threshold = abs(rand())
    max_iterations = rand(1:100) 

    let x0 = x0, long_stepsize = long_stepsize, α_lower = α_lower, 
        α_default = α_default, init_stepsize = init_stepsize, 
        δ0 = δ0, δ_upper = δ_upper, ρ = ρ, M = M, threshold = threshold,
        max_iterations = max_iterations

        # random field values -- error: δ0 < 0
        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = -δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)
        
        # random field values -- error δ_upper > δ0
        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0 - 1,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)

        # random field values -- error: α_lower <= 0
        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = 0.0,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)

        # random field values -- error: α_lower <= 0
        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = -1.0,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)

        # random field values -- error: α_default <= 0
        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = 0.0,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)
        
        # random field values -- error: α_default <= 0
        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = -1.0,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)

        # random field values -- error: init_stepsize outside interval
        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = -1.0, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)

    end
end

@testset "Utility -- Update Algorithm Parameters (GDNonseqArmijoBB)" begin

    include("../utility/update_algorithm_parameters_test_cases.jl")

    # arguments
    dim = 50
    x0 = randn(dim)
    long_stepsize = true
    α_lower = abs(rand())
    α_default = α_lower + 1
    init_stepsize = (α_lower + 1/α_lower)/2
    δ0 = abs(randn())
    δ_upper = δ0 + 1
    ρ = abs(randn())
    M = rand(1:100)
    threshold = abs(randn())
    max_iterations = rand(1:100)

    # build structure
    optData = NonsequentialArmijoSafeBBGD(Float64;
        x0 = x0, init_stepsize = init_stepsize, long_stepsize = long_stepsize,
        α_lower = α_lower, α_default = α_default, δ0 = δ0, δ_upper = δ_upper,
        ρ = ρ, M = M, threshold = threshold, max_iterations = max_iterations)

    # Conduct test cases
    update_algorithm_parameters_test_cases(optData, dim, max_iterations)
end

@testset "Utility -- Inner Loop (GDNonseqArmijoBB)" begin

    ## Inner loop with long_stepsize = true
    dim = 50
    x0 = randn(dim)
    long_stepsize = true
    α_lower = abs(rand())
    α_default = α_lower + 1
    init_stepsize = (α_lower + 1/α_lower)/2
    δ0 = abs(randn())
    δ_upper = δ0 + 1
    ρ = abs(randn())
    M = rand(1:100)
    threshold = abs(randn())
    max_iterations = rand(1:100)

    ## build structure
    optData = NonsequentialArmijoSafeBBGD(Float64; 
        x0 = x0,
        init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, 
        α_lower = α_lower,
        α_default = α_default,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ, 
        M = M,
        threshold = threshold,
        max_iterations = max_iterations)

    progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)
    precomp, store = OptimizationMethods.initialize(progData)
    ks = [1, max_iterations]

    for k in ks
        # Test first event trigger: radius violation
        let ψjk=x0 .+ 11, θk=x0, optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 1.5
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k, max_iteration=100)

            @test ψjk == x0 .+ 11
        end

        # Test second event trigger: τ_lower 
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 0.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k, max_iteration=100)

            @test ψjk == x0
        end

        # Test third event trigger: τ_upper 
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 2.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k, max_iteration=100)

            @test ψjk == x0
        end

        # Test fourth event trigger: max_iteration
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 1.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k, max_iteration=0)

            @test ψjk == x0
        end

        # Test first iteration -- second_acceptance_occurred = false
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            j=1

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.second_acceptance_occurred = false

            optData.δk = 1.5

            α = optData.init_stepsize
            if α < optData.α_lower || α > (1/optData.α_lower)
                α = optData.α_default
            end
            step = (optData.δk * α) .* store.grad 

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = 1
            )

            @test ψjk == θk - step
            @test optData.iter_diff ≈ -step 
            @test optData.grad_diff ≈ OptimizationMethods.grad(progData, ψjk) -
                OptimizationMethods.grad(progData, θk)
            @test optData.α0k == α
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ == norm(store.grad)
            @test optData.second_acceptance_occurred == false
        end

        # Test first iteration -- second_acceptance_occurred = true
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            j=1

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)

            # check to make sure α0k is the bb step size
            optData.iter_diff = randn(length(x0))
            optData.grad_diff = randn(length(x0))
            optData.second_acceptance_occurred = true

            optData.δk = 1.5

            α = optData.bb_step_size(optData.iter_diff, optData.grad_diff)
            if α < optData.α_lower || α > (1/optData.α_lower)
                α = optData.α_default
            end
            step = (optData.δk * α) .* store.grad 

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = 1
            )

            @test ψjk == θk - step
            @test optData.iter_diff ≈ -step 
            @test optData.grad_diff ≈ OptimizationMethods.grad(progData, ψjk) -
                OptimizationMethods.grad(progData, θk) 
            @test optData.α0k == α
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ == norm(store.grad)
        end

        # Test random iteration -- second_acceptance_occurred = false
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            #To do this test correctly, we would need to know at what iteration 
            #j an inner loop exists.
            max_iteration = rand(2:100)

            # Reset 
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1e-10
            optData.second_acceptance_occurred = false

            #Get exit iteration j
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = max_iteration
            )

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1e-10 
            optData.second_acceptance_occurred = false

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = j-1
            )
            α = 0.0
            if j == 1
                α = optData.init_stepsize
                if α < optData.α_lower || α > (1/optData.α_lower)
                    α = optData.α_default
                end
            else
                α = optData.bb_step_size(optData.iter_diff, optData.grad_diff)
                if α < optData.α_lower || α > (1/optData.α_lower)
                    α = optData.α_default
                end
            end

            ψ_jm1_k = copy(ψjk)
            grd = OptimizationMethods.grad(progData, ψ_jm1_k)
            step = (optData.δk * α) * grd

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1e-10
            optData.second_acceptance_occurred = false

            #Get ψ_{j,k}
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = max_iteration
            )

            @test ψjk ≈ ψ_jm1_k - step
            @test optData.iter_diff ≈ -step 
            @test optData.grad_diff ≈ OptimizationMethods.grad(progData, ψjk) -
                OptimizationMethods.grad(progData, ψ_jm1_k)
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ ≈ norm(store.grad)

            α = optData.init_stepsize
            if α < optData.α_lower || α > (1/optData.α_lower)
                α = optData.α_default
            end
            @test optData.α0k == α
        end

        # Test random iteration -- second_acceptance_occurred = false
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            #To do this test correctly, we would need to know at what iteration 
            #j an inner loop exists.
            max_iteration = rand(2:100)
            iter_diff = randn(length(x0))
            grad_diff = randn(length(x0))

            # Reset 
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1e-10
            optData.iter_diff = iter_diff
            optData.grad_diff = grad_diff
            optData.second_acceptance_occurred = true

            #Get exit iteration j
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = max_iteration
            )

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1e-10 
            optData.iter_diff = iter_diff
            optData.grad_diff = grad_diff
            optData.second_acceptance_occurred = true

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = j-1
            )
            α = optData.bb_step_size(optData.iter_diff, optData.grad_diff)
            if α < optData.α_lower || α > (1/optData.α_lower)
                α = optData.α_default
            end

            ψ_jm1_k = copy(ψjk)
            grd = OptimizationMethods.grad(progData, ψ_jm1_k)
            step = (optData.δk * α) * grd

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1e-10
            optData.iter_diff = iter_diff
            optData.grad_diff = grad_diff
            optData.second_acceptance_occurred = true

            #Get ψ_{j,k}
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = max_iteration
            )

            @test ψjk ≈ ψ_jm1_k - step
            @test optData.iter_diff ≈ -step 
            @test optData.grad_diff ≈ OptimizationMethods.grad(progData, ψjk) -
                OptimizationMethods.grad(progData, ψ_jm1_k) rtol = 1e-7
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ ≈ norm(store.grad)

            α0k = optData.bb_step_size(iter_diff, grad_diff)
            if α0k < optData.α_lower || α0k > (1/optData.α_lower)
                α0k = optData.α_default
            end
            @test optData.α0k == α0k
        end

    end
end

@testset "Method -- GD with Nonsequential Armijo and BB Steps: method (monotone)" begin

    # Default Parameters
    dim = 50
    x0 = randn(dim)
    long_stepsize = true
    α_lower = abs(rand())
    α_default = α_lower + 1
    init_stepsize = (α_lower + 1/α_lower)/2
    δ0 = abs(randn())
    δ_upper = δ0 + 1
    ρ = abs(randn())
    M = 1 
    threshold = abs(randn())
    max_iterations = rand(1:100)

    # Should exit on iteration 0 because max_iterations is 0
    let x0=copy(x0), δ0=δ0, δ_upper=δ_upper, ρ=ρ, threshold=threshold, 
        max_iterations=0

        # Specify optimization method and problem
        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # Run method
        x = nonsequential_armijo_safe_bb_gd(optData, progData)

        @test optData.stop_iteration == 0
        @test progData.counters.neval_obj == 1 
        @test progData.counters.neval_grad == 1
        @test x == x0
        
        grd = OptimizationMethods.grad(progData, x0)
        grd_norm = norm(grd)
        @test optData.grad_val_hist ≈ [norm(grd)]
        @test optData.τ_lower ≈ norm(grd) / sqrt(2)
        @test optData.τ_upper ≈ norm(grd) * sqrt(10)
    end

    # should exit on iteration 0 because threshold is larger than gradient 
    let x0=copy(x0), δ0=δ0, δ_upper=δ_upper, ρ=ρ, threshold=1e4, 
        max_iterations=max_iterations

        # Specify optimization method and problem
        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # Run method
        x = nonsequential_armijo_safe_bb_gd(optData, progData)

        @test optData.stop_iteration == 0
        @test progData.counters.neval_obj == 1 
        @test progData.counters.neval_grad == 1
        @test x == x0
        
        grd = OptimizationMethods.grad(progData, x0)
        grd_norm = norm(grd)
        @test optData.grad_val_hist[1] ≈ norm(grd)
        @test optData.τ_lower ≈ norm(grd) / sqrt(2)
        @test optData.τ_upper ≈ norm(grd) * sqrt(10)
    end
    
    # test beyond the first iteration 
    factor = 1
    let x0=copy(x0), δ0=factor * δ0, δ_upper=factor * δ_upper, ρ=ρ, threshold=threshold, 
        max_iterations=100

        #Specify Problem 
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # Specify optimization method for exit_iteration - 1
        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)
        
        x = nonsequential_armijo_safe_bb_gd(optData, progData)
        g = OptimizationMethods.grad(progData, x)
        
        stop_iteration = optData.stop_iteration 

        @test optData.iter_hist[stop_iteration+1] == x
        @test optData.grad_val_hist[stop_iteration+1] ≈ norm(g)

        # Find the first time we accept an iterate, stop 1 before it
        # and verify that the inner loop is correct
        first_acceptance = 1
        while (first_acceptance < stop_iteration) && 
                (optData.iter_hist[first_acceptance] == 
                    optData.iter_hist[first_acceptance + 1])
            first_acceptance += 1
        end

        ## reset
        precomp, store = OptimizationMethods.initialize(progData)
        F(x) = OptimizationMethods.obj(progData, x)
        for k in 1:(first_acceptance-1)

            # create optdata for k - 1 and k
            optDatakm1 = NonsequentialArmijoSafeBBGD(Float64; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoSafeBBGD(Float64; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = k) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_safe_bb_gd(optDatakm1, progData)  
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, k)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), optDatakm1.reference_value, 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_safe_bb_gd(optDatak, progData)

            ## check gradient quantities
            @test isapprox(optDatak.∇F_θk, 
                OptimizationMethods.grad(progData, xkm1))

            ## Check that that the algorithm updated the parameters correctly
            @test optDatak.δk == (optDatakm1.δk * .5)
            @test xk == xkm1

            ## test field values at time k
            @test optDatak.reference_value == maximum(optDatak.objective_hist)
            @test optDatak.objective_hist[optDatak.reference_value_index] ==
                optDatak.reference_value
            @test optDatak.iter_hist[k+1] == xk
            @test optDatak.grad_val_hist[k+1] == optDatak.grad_val_hist[k]
            @test optDatak.grad_val_hist[k+1] ≈ 
                norm(OptimizationMethods.grad(progData, xkm1))
            @test optDatakm1.second_acceptance_occurred == false
            @test optDatak.second_acceptance_occurred == false
            @test optDatak.α0k == optDatak.init_stepsize
            @test optDatak.iter_diff ≈ optDatakm1.iter_diff_checkpoint
            @test optDatak.grad_diff ≈ optDatakm1.iter_diff_checkpoint
        end 

        # Test values are correctly updated for acceptance
        iter = first_acceptance - 1

        # create optdata for k - 1 and k
        optDatakm1 = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = iter) ## stop_iteration = iter

        optDatak = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = iter + 1) ## stop_iteration = iter + 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_safe_bb_gd(optDatakm1, progData)  
        xk = nonsequential_armijo_safe_bb_gd(optDatak, progData)

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, iter + 1)

        # test that non sequential armijo condition is accepted  
        achieved_descent = OptimizationMethods.non_sequential_armijo_condition(
            F(x), optDatakm1.reference_value, optDatakm1.grad_val_hist[iter + 1], 
            optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)
        @test achieved_descent
        
        # Update the parameters in optDatakm1
        flag = OptimizationMethods.update_algorithm_parameters!(x, optDatakm1, 
            achieved_descent, iter + 1)

        # update the cache at time k - 1
        optDatakm1.acceptance_cnt += 1
        optDatakm1.objective_hist[optDatakm1.acceptance_cnt] = F(x)
        if ((optDatakm1.acceptance_cnt - 1) % M) + 1 == optDatakm1.reference_value_index
            optDatakm1.reference_value, optDatakm1.reference_value_index =
            findmax(optDatakm1.objective_hist)
        end
        
        # Check that optDatak matches optDatakm1
        @test flag
        @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
        @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
        @test optDatak.δk == optDatakm1.δk
        @test optDatak.τ_lower == optDatakm1.τ_lower
        @test optDatak.τ_upper == optDatakm1.τ_upper
        @test xk == x

        ## test field values at time k
        @test optDatak.reference_value == maximum(optDatak.objective_hist)
        @test optDatak.objective_hist[optDatak.reference_value_index] ==
            optDatak.reference_value
        @test optDatak.reference_value == optDatakm1.reference_value
        @test optDatak.iter_hist[iter+1] == xkm1
        @test optDatak.iter_hist[iter+2] == xk
        @test optDatak.grad_val_hist[iter+2] ≈ 
            norm(OptimizationMethods.grad(progData, x))
        @test optDatak.second_acceptance_occurred == true
        @test optDatak.α0k == optDatak.init_stepsize
        @test optDatak.iter_diff ≈ optDatakm1.iter_diff
        @test optDatak.grad_diff ≈ optDatakm1.grad_diff
        
        # Find window of non-accepted iterates for last accepted iterate
        last_acceptance = stop_iteration - 1
        while (1 < last_acceptance) && 
                (optData.iter_hist[last_acceptance] == 
                    optData.iter_hist[last_acceptance + 1])
            last_acceptance -= 1
        end
        last_acceptance += 1

        for k in last_acceptance:(stop_iteration-1)

            # create optdata for k - 1 and k
            optDatakm1 = NonsequentialArmijoSafeBBGD(Float64; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoSafeBBGD(Float64; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = k) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_safe_bb_gd(optDatakm1, progData)  
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, k)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), optDatakm1.reference_value, 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_safe_bb_gd(optDatak, progData)

            ## check gradient quantities
            @test isapprox(optDatak.∇F_θk, 
                OptimizationMethods.grad(progData, xkm1))

            ## Check that that the algorithm updated the parameters correctly
            @test optDatak.δk == (optDatakm1.δk * .5)
            @test xk == xkm1

            ## test field values at time k
            @test optDatak.reference_value == maximum(optDatak.objective_hist)
            @test optDatak.objective_hist[optDatak.reference_value_index] ==
                optDatak.reference_value
            @test optDatak.iter_hist[k+1] == xk
            @test optDatak.grad_val_hist[k+1] == optDatak.grad_val_hist[k]
            @test optDatak.grad_val_hist[k+1] ≈ 
                norm(OptimizationMethods.grad(progData, xkm1))
            @test optDatak.second_acceptance_occurred == true

            α = optDatak.bb_step_size(optDatak.iter_diff_checkpoint,
                optDatak.grad_diff_checkpoint)
            if α < optData.α_lower || α > (1/optData.α_lower)
                α = optData.α_default
            end
            @test optDatak.α0k ≈ α

            @test optDatak.iter_diff ≈ optDatakm1.iter_diff_checkpoint
            @test optDatak.grad_diff ≈ optDatakm1.grad_diff_checkpoint
        end

        # Test values are correctly updated for acceptance
        iter = stop_iteration - 1

        # create optdata for k - 1 and k
        optDatakm1 = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = iter) ## stop_iteration = iter

        optDatak = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = iter + 1) ## stop_iteration = iter + 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_safe_bb_gd(optDatakm1, progData)  
        xk = nonsequential_armijo_safe_bb_gd(optDatak, progData)

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, iter + 1)

        # test that non sequential armijo condition is accepted  
        achieved_descent = OptimizationMethods.non_sequential_armijo_condition(
            F(x), F(xkm1), optDatakm1.grad_val_hist[iter + 1], 
            optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)
        @test achieved_descent
        
        # Update the parameters in optDatakm1
        flag = OptimizationMethods.update_algorithm_parameters!(x, optDatakm1, 
            achieved_descent, iter + 1)

        # update the cache at time k - 1
        optDatakm1.acceptance_cnt += 1
        optDatakm1.objective_hist[optDatakm1.acceptance_cnt] = F(x)
        if ((optDatakm1.acceptance_cnt-1) % M) + 1 == optDatakm1.reference_value_index
            optDatakm1.reference_value, optDatakm1.reference_value_index =
            findmax(optDatakm1.objective_hist)
        end
                
        # Check that optDatak matches optDatakm1
        @test flag
        @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
        @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
        @test optDatak.δk == optDatakm1.δk
        @test optDatak.τ_lower == optDatakm1.τ_lower
        @test optDatak.τ_upper == optDatakm1.τ_upper
        @test xk == x

        ## test field values at time k
        @test optDatak.reference_value == maximum(optDatak.objective_hist)
        @test optDatak.objective_hist[optDatak.reference_value_index] ==
            optDatak.reference_value
        @test optDatak.reference_value == optDatakm1.reference_value
        @test optDatak.iter_hist[iter+1] == xkm1
        @test optDatak.iter_hist[iter+2] == xk
        @test optDatak.grad_val_hist[iter+2] ≈ 
            norm(OptimizationMethods.grad(progData, x))
        @test optDatak.second_acceptance_occurred == true

        α = optDatak.bb_step_size(optDatak.iter_diff_checkpoint,
        optDatak.grad_diff_checkpoint)
        if α < optData.α_lower || α > (1/optData.α_lower)
            α = optData.α_default
        end
        @test optDatak.α0k ≈ α

        @test optDatak.iter_diff ≈ optDatakm1.iter_diff
        @test optDatak.grad_diff ≈ optDatakm1.grad_diff

    end

end

@testset "Method -- GD with Nonsequential Armijo and BB Steps: method (nonmonotone)" begin

    # Default Parameters
    dim = 50
    x0 = randn(dim)
    long_stepsize = true
    α_lower = abs(rand())
    α_default = α_lower + 1
    init_stepsize = (α_lower + 1/α_lower)/2
    δ0 = rand()
    δ_upper = 1.0
    ρ = rand()
    M = rand(3:10)
    threshold = 1e-10
    max_iterations = 100

    # Should exit on iteration 0 because max_iterations is 0
    let x0=copy(x0), δ0=δ0, δ_upper=δ_upper, ρ=ρ, threshold=threshold, 
        max_iterations=0

        # Specify optimization method and problem
        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # Run method
        x = nonsequential_armijo_safe_bb_gd(optData, progData)

        @test optData.stop_iteration == 0
        @test progData.counters.neval_obj == 1 
        @test progData.counters.neval_grad == 1
        @test x == x0
        
        grd = OptimizationMethods.grad(progData, x0)
        grd_norm = norm(grd)
        @test optData.grad_val_hist ≈ [norm(grd)]
        @test optData.τ_lower ≈ norm(grd) / sqrt(2)
        @test optData.τ_upper ≈ norm(grd) * sqrt(10)
    end

    # should exit on iteration 0 because threshold is larger than gradient 
    let x0=copy(x0), δ0=δ0, δ_upper=δ_upper, ρ=ρ, threshold=1e4, 
        max_iterations=max_iterations

        # Specify optimization method and problem
        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # Run method
        x = nonsequential_armijo_safe_bb_gd(optData, progData)

        @test optData.stop_iteration == 0
        @test progData.counters.neval_obj == 1 
        @test progData.counters.neval_grad == 1
        @test x == x0
        
        grd = OptimizationMethods.grad(progData, x0)
        grd_norm = norm(grd)
        @test optData.grad_val_hist[1] ≈ norm(grd)
        @test optData.τ_lower ≈ norm(grd) / sqrt(2)
        @test optData.τ_upper ≈ norm(grd) * sqrt(10)
    end
    
    # test beyond the first iteration 
    factor = 1
    let x0=copy(x0), δ0=factor * δ0, δ_upper=factor * δ_upper, ρ=ρ, threshold=threshold, 
        max_iterations=100

        #Specify Problem 
        progData = OptimizationMethods.LogisticRegression(Float64, nvar=dim)

        # Specify optimization method for exit_iteration - 1
        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)
        
        x = nonsequential_armijo_safe_bb_gd(optData, progData)
        g = OptimizationMethods.grad(progData, x)
        
        stop_iteration = optData.stop_iteration 

        @test optData.iter_hist[stop_iteration+1] == x
        @test optData.grad_val_hist[stop_iteration+1] ≈ norm(g)

        # Find the first time we accept an iterate, stop 1 before it
        # and verify that the inner loop is correct
        first_acceptance = 1
        while (first_acceptance < stop_iteration) && 
                (optData.iter_hist[first_acceptance] == 
                    optData.iter_hist[first_acceptance + 1])
            first_acceptance += 1
        end

        ## reset
        precomp, store = OptimizationMethods.initialize(progData)
        F(x) = OptimizationMethods.obj(progData, x)
        for k in 1:(first_acceptance-1)

            # create optdata for k - 1 and k
            optDatakm1 = NonsequentialArmijoSafeBBGD(Float64; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoSafeBBGD(Float64; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = k) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_safe_bb_gd(optDatakm1, progData)  
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, k)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), optDatakm1.reference_value, 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_safe_bb_gd(optDatak, progData)

            ## check gradient quantities
            @test isapprox(optDatak.∇F_θk, 
                OptimizationMethods.grad(progData, xkm1))

            ## Check that that the algorithm updated the parameters correctly
            @test optDatak.δk == (optDatakm1.δk * .5)
            @test xk == xkm1

            ## test field values at time k
            @test optDatak.reference_value == maximum(optDatak.objective_hist)
            @test optDatak.objective_hist[optDatak.reference_value_index] ==
                optDatak.reference_value
            @test optDatak.iter_hist[k+1] == xk
            @test optDatak.grad_val_hist[k+1] == optDatak.grad_val_hist[k]
            @test optDatak.grad_val_hist[k+1] ≈ 
                norm(OptimizationMethods.grad(progData, xkm1))
            @test optDatakm1.second_acceptance_occurred == false
            @test optDatak.second_acceptance_occurred == false
            @test optDatak.α0k == optDatak.init_stepsize
            @test optDatak.iter_diff ≈ optDatakm1.iter_diff_checkpoint
            @test optDatak.grad_diff ≈ optDatakm1.iter_diff_checkpoint
        end 

        # Test values are correctly updated for acceptance
        iter = first_acceptance - 1

        # create optdata for k - 1 and k
        optDatakm1 = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = iter) ## stop_iteration = iter

        optDatak = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = iter + 1) ## stop_iteration = iter + 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_safe_bb_gd(optDatakm1, progData)  
        xk = nonsequential_armijo_safe_bb_gd(optDatak, progData)

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, iter + 1)

        # test that non sequential armijo condition is accepted  
        achieved_descent = OptimizationMethods.non_sequential_armijo_condition(
            F(x), optDatakm1.reference_value, optDatakm1.grad_val_hist[iter + 1], 
            optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)
        @test achieved_descent
        
        # Update the parameters in optDatakm1
        flag = OptimizationMethods.update_algorithm_parameters!(x, optDatakm1, 
            achieved_descent, iter + 1)

        # update the cache at time k - 1
        optDatakm1.acceptance_cnt += 1
        optDatakm1.objective_hist[optDatakm1.acceptance_cnt] = F(x)
        if ((optDatakm1.acceptance_cnt-1) % M) + 1 == optDatakm1.reference_value_index
            optDatakm1.reference_value, optDatakm1.reference_value_index =
            findmax(optDatakm1.objective_hist)
        end
        

        # Check that optDatak matches optDatakm1
        @test flag
        @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
        @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
        @test optDatak.δk == optDatakm1.δk
        @test optDatak.τ_lower == optDatakm1.τ_lower
        @test optDatak.τ_upper == optDatakm1.τ_upper
        @test xk == x

        ## test field values at time k
        @test optDatak.reference_value == maximum(optDatak.objective_hist)
        @test optDatak.objective_hist[optDatak.reference_value_index] ==
            optDatak.reference_value
        @test optDatak.reference_value == optDatakm1.reference_value
        @test optDatak.iter_hist[iter+1] == xkm1
        @test optDatak.iter_hist[iter+2] == xk
        @test optDatak.grad_val_hist[iter+2] ≈ 
            norm(OptimizationMethods.grad(progData, x))
        @test optDatak.second_acceptance_occurred == true
        @test optDatak.α0k == optDatak.init_stepsize
        @test optDatak.iter_diff ≈ optDatakm1.iter_diff
        @test optDatak.grad_diff ≈ optDatakm1.grad_diff
        
        # Find window of non-accepted iterates for last accepted iterate
        last_acceptance = stop_iteration - 1
        while (1 < last_acceptance) && 
                (optData.iter_hist[last_acceptance] == 
                    optData.iter_hist[last_acceptance + 1])
            last_acceptance -= 1
        end
        last_acceptance += 1

        for k in last_acceptance:(stop_iteration-1)

            # create optdata for k - 1 and k
            optDatakm1 = NonsequentialArmijoSafeBBGD(Float64; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoSafeBBGD(Float64; 
                x0 = x0,
                init_stepsize = init_stepsize, 
                long_stepsize = long_stepsize, 
                α_lower = α_lower,
                α_default = α_default,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ, 
                M = M,
                threshold = threshold,
                max_iterations = k) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_safe_bb_gd(optDatakm1, progData)  
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, k)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), optDatakm1.reference_value, 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_safe_bb_gd(optDatak, progData)

            ## check gradient quantities
            @test isapprox(optDatak.∇F_θk, 
                OptimizationMethods.grad(progData, xkm1))

            ## Check that that the algorithm updated the parameters correctly
            @test optDatak.δk == (optDatakm1.δk * .5)
            @test xk == xkm1

            ## test field values at time k
            @test optDatak.reference_value == maximum(optDatak.objective_hist)
            @test optDatak.objective_hist[optDatak.reference_value_index] ==
                optDatak.reference_value
            @test optDatak.iter_hist[k+1] == xk
            @test optDatak.grad_val_hist[k+1] == optDatak.grad_val_hist[k]
            @test optDatak.grad_val_hist[k+1] ≈ 
                norm(OptimizationMethods.grad(progData, xkm1))
            @test optDatak.second_acceptance_occurred == true

            α = optDatak.bb_step_size(optDatak.iter_diff_checkpoint,
                optDatak.grad_diff_checkpoint)
            if α < optData.α_lower || α > (1/optData.α_lower)
                α = optData.α_default
            end
            @test optDatak.α0k ≈ α
            
            @test optDatak.iter_diff ≈ optDatakm1.iter_diff_checkpoint
            @test optDatak.grad_diff ≈ optDatakm1.grad_diff_checkpoint
        end

        # Test values are correctly updated for acceptance
        iter = last_acceptance - 2

        # create optdata for k - 1 and k
        optDatakm1 = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = iter) ## stop_iteration = iter = last_acceptance - 2

        optDatak = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_default = α_default,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = iter + 1) ## stop_iteration = iter + 1 = last_acceptance - 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_safe_bb_gd(optDatakm1, progData)  
        xk = nonsequential_armijo_safe_bb_gd(optDatak, progData)
        @test xk != xkm1

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, iter + 1)

        # test that non sequential armijo condition is accepted  
        achieved_descent = OptimizationMethods.non_sequential_armijo_condition(
            F(x), F(xkm1), optDatakm1.grad_val_hist[iter + 1], 
            optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)
        @test achieved_descent
        
        # Update the parameters in optDatakm1
        flag = OptimizationMethods.update_algorithm_parameters!(x, optDatakm1, 
            achieved_descent, iter + 1)

        # update the cache at time k - 1
        optDatakm1.acceptance_cnt += 1
        optDatakm1.objective_hist[optDatakm1.acceptance_cnt] = F(x)
        if ((optDatakm1.acceptance_cnt-1) % M) + 1 == optDatakm1.reference_value_index
            optDatakm1.reference_value, optDatakm1.reference_value_index =
            findmax(optDatakm1.objective_hist)
        end
        
        # Check that optDatak matches optDatakm1
        @test flag
        @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
        @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
        @test optDatak.δk == optDatakm1.δk
        @test optDatak.τ_lower == optDatakm1.τ_lower
        @test optDatak.τ_upper == optDatakm1.τ_upper
        @test xk == x

        ## test field values at time k
        @test optDatak.reference_value == maximum(optDatak.objective_hist)
        @test optDatak.objective_hist[optDatak.reference_value_index] ==
            optDatak.reference_value
        @test optDatak.reference_value == optDatakm1.reference_value
        @test optDatak.iter_hist[iter+1] == xkm1
        @test optDatak.iter_hist[iter+2] == xk
        @test optDatak.grad_val_hist[iter+2] ≈ 
            norm(OptimizationMethods.grad(progData, x))
        @test optDatak.second_acceptance_occurred == true

        α = optDatak.bb_step_size(optDatak.iter_diff_checkpoint,
        optDatak.grad_diff_checkpoint)
        if α < optData.α_lower || α > (1/optData.α_lower)
            α = optData.α_default
        end
        @test optDatak.α0k ≈ α

        @test optDatak.iter_diff ≈ optDatakm1.iter_diff
        @test optDatak.grad_diff ≈ optDatakm1.grad_diff

    end

end

end # End module