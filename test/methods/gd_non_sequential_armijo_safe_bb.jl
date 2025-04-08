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
    unique_fields = [:∇F_θk, :norm_∇F_ψ, :init_stepsize, :bb_step_size,
        :α0k, :α_lower, :α_upper, :iter_diff_checkpoint, :grad_diff_checkpoint,
        :iter_diff, :grad_diff, :δk, :δ_upper, :ρ, :objective_hist,
        :reference_value, :reference_value_index, :τ_lower, :τ_upper,
        :second_acceptance_occurred]
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
        [:α_upper, type],
        [:iter_diff_checkpoint, Vector{type}],
        [:grad_diff_checkpoint, Vector{type}],
        [:iter_diff, Vector{type}],
        [:grad_diff, Vector{type}],
        [:δk, type],
        [:δ_upper, type],
        [:ρ, type],
        [:objective_hist, Vector{type}],
        [:reference_value, type],
        [:reference_value_index, Int64],
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
            α_upper = α_lower + 1
            init_stepsize = (α_lower + α_upper)/2
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
                α_upper = α_upper,
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
            α_upper = α_lower + 1
            init_stepsize = (α_lower + α_upper)/2
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
                α_upper = α_upper,
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

    # test struct initializations
    let

        # random field values -- long_stepsize = true
        x0 = randn(10)
        long_stepsize = true
        α_lower = abs(rand())
        α_upper = α_lower + 1
        init_stepsize = (α_lower + α_upper)/2
        δ0 = abs(rand())
        δ_upper = δ0 + 1
        ρ = rand()
        M = rand(1:100)
        threshold = abs(rand())
        max_iterations = rand(1:100) 

        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_upper = α_upper,
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
        @test optData.α_upper == α_upper
        @test optData.δk == δ0
        @test optData.δ_upper == δ_upper
        @test optData.ρ == ρ
        @test length(optData.objective_hist) == M
        @test optData.threshold == threshold
        @test optData.max_iterations == max_iterations
        
        # random field values -- long_stepsize = false
        x0 = randn(10)
        long_stepsize = false
        α_lower = abs(rand())
        α_upper = α_lower + 1
        init_stepsize = (α_upper + α_lower)/2
        δ0 = abs(rand())
        δ_upper = δ0 + 1
        ρ = rand()
        M = rand(1:100)
        threshold = abs(rand())
        max_iterations = rand(1:100)

        optData = NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_upper = α_upper,
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
        @test optData.α_upper == α_upper
        @test optData.δk == δ0
        @test optData.δ_upper == δ_upper
        @test optData.ρ == ρ
        @test length(optData.objective_hist) == M
        @test optData.threshold == threshold
        @test optData.max_iterations == max_iterations
        @test optData.second_acceptance_occurred == false

    end 

    # test struct error
    let

        # random field values -- error: δ0 < 0
        x0 = randn(10)
        long_stepsize = rand([true, false])
        α_lower = abs(rand())
        α_upper = α_lower + 1
        init_stepsize = (α_upper + α_lower)/2
        δ0 = -abs(rand())                       # δ < 0
        δ_upper = δ0 + 1
        ρ = rand()
        M = rand(1:100)
        threshold = abs(rand())
        max_iterations = rand(1:100) 

        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_upper = α_upper,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)
        
        # random field values -- error δ_upper > δ0
        x0 = randn(10)
        long_stepsize = false
        α_lower = abs(rand())
        α_upper = α_lower + 1
        init_stepsize = (α_upper + α_lower)/2
        δ0 = abs(rand())
        δ_upper = δ0 - 1                        # δ_upper < δ0
        ρ = rand()
        M = rand(1:100)
        threshold = abs(rand())
        max_iterations = rand(1:100)

        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_upper = α_upper,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)

        # random field values -- error: α_lower <= 0
        x0 = randn(10)
        long_stepsize = false
        α_lower = 0.0
        α_upper = α_lower + 1
        init_stepsize = (α_upper + α_lower)/2
        δ0 = abs(rand())
        δ_upper = δ0 + 1
        ρ = rand()
        M = rand(1:100)
        threshold = abs(rand())
        max_iterations = rand(1:100)

        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_upper = α_upper,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)

        # random field values -- error: α_lower <= 0
        x0 = randn(10)
        long_stepsize = false
        α_lower = -1.0
        α_upper = α_lower + 1
        init_stepsize = (α_upper + α_lower)/2
        δ0 = abs(rand())
        δ_upper = δ0 + 1
        ρ = rand()
        M = rand(1:100)
        threshold = abs(rand())
        max_iterations = rand(1:100)

        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_upper = α_upper,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)

        # random field values -- error: α_lower > α_upper
        x0 = randn(10)
        long_stepsize = false
        α_lower = abs(rand())
        α_upper = α_lower - 1
        init_stepsize = (α_upper + α_lower)/2
        δ0 = abs(rand())
        δ_upper = δ0 + 1
        ρ = rand()
        M = rand(1:100)
        threshold = abs(rand())
        max_iterations = rand(1:100)

        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_upper = α_upper,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ, 
            M = M,
            threshold = threshold,
            max_iterations = max_iterations)

        # random field values -- error: init_stepsize outside interval
        x0 = randn(10)
        long_stepsize = false
        α_lower = abs(rand())
        α_upper = α_lower + 1
        init_stepsize = -1.0
        δ0 = abs(rand())
        δ_upper = δ0 + 1
        ρ = rand()
        M = rand(1:100)
        threshold = abs(rand())
        max_iterations = rand(1:100)

        @test_throws AssertionError NonsequentialArmijoSafeBBGD(Float64; 
            x0 = x0,
            init_stepsize = init_stepsize, 
            long_stepsize = long_stepsize, 
            α_lower = α_lower,
            α_upper = α_upper,
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

    ## Inner loop with long_stepsize = true
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

    ## build structure
    optData = NonsequentialArmijoSafeBBGD(Float64; 
        x0 = x0,
        init_stepsize = init_stepsize, 
        long_stepsize = long_stepsize, 
        α_lower = α_lower,
        α_upper = α_upper,
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
            α = min(max(α, optData.α_lower), optData.α_upper)
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
            α = min(max(α, optData.α_lower), optData.α_upper)
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
                α = min(max(α, optData.α_lower), optData.α_upper)
            else
                α = optData.bb_step_size(optData.iter_diff, optData.grad_diff)
                α = min(max(α, optData.α_lower), optData.α_upper)
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
            @test optData.α0k == min(max(optData.init_stepsize, 
                optData.α_lower), optData.α_upper)
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
            α = min(max(α, optData.α_lower), optData.α_upper)

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
                OptimizationMethods.grad(progData, ψ_jm1_k)
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ ≈ norm(store.grad)

            α0k = optData.bb_step_size(iter_diff, grad_diff)
            @test optData.α0k == min(max(α0k, optData.α_lower), optData.α_upper)
        end

    end
end

@testset "Method -- GD with Nonsequential Armijo and BB Steps: method" begin
end

end # End module