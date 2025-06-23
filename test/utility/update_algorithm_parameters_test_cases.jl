# Date: 2025/19/03
# Author: Christian Varner
# Purpose: Testing function for update_algorithm_parameters in
# src/methods/non_sequential_armijo.jl

using OptimizationMethods, Test

"""
    update_algorithm_parameters_test_cases(optData::P 
        where P <: OptimizationMethods.AbstractOptimizerData{T}, dim::Int64,
        max_iterations::Int64; constant_fields::Vector{Symbol} = 
        Vector{Symbol}()) where {T}

Test cases for updating the parameters of a nonsequential armijo method.

# Arguments

- `optData::P where P <: OptimizationMethods.AbstractOptimizerData{T}`, `struct`
    that specifies the method
- `dim::Int64`, length of the value of the keyword argument `x0` used to initialize
    the `struct` `optData`. Used for testing.
- `max_iterations::Int64`, value that was used to initialize the `struct` `optData`.
    Used for testing.

## Keyword Arguments

- `constant_fields::Vector{Symbol} = Vector{Symbol}()`, the fields that should
    remain constant after this operation, and that need to be tested for such
    property.

# Returns

Nothing
"""
function update_algorithm_parameters_test_cases(optData::P 
    where P <: OptimizationMethods.AbstractOptimizerData{T}, dim::Int64,
    max_iterations::Int64; constant_fields::Vector{Symbol} = Vector{Symbol}()
    ) where {T}
    
    ############################################################################
    # Case 1: Did not satisfy armijo condition
    ############################################################################
    let optData = optData, achieved_descent = false, dim = dim
        # First Iteration
        xp1 = zeros(dim)
        iter = 1
        optData.τ_lower = 0.0
        optData.τ_upper = 1.0
        optData.δk = 1.0

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 == optData.iter_hist[iter]
        @test optData.τ_lower == 0.0 
        @test optData.τ_upper == 1.0
        @test optData.δk == 0.5
        @test !params_update_flag
        
        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

    end

    let optData = optData, achieved_descent = false, dim = dim,
        max_iterations = max_iterations 
        # General Iteration 
        xp1 = zeros(dim)
        iter = rand(3:max_iterations)
        optData.τ_lower = 0.0 
        optData.τ_upper = 1.0 
        optData.δk = 1.0 

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        optData.iter_hist[iter] = rand(dim)
        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test iszero(xp1 - optData.iter_hist[iter]) 
        @test optData.τ_lower == 0.0 
        @test optData.τ_upper == 1.0
        @test optData.δk == 0.5
        @test !params_update_flag
        
    end

    ############################################################################
    # Case 2: Did satisfy condition + grad-norm smaller than lower bound
    ############################################################################
    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 0.5
        optData.δk = 1.0

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 0.5 / sqrt(2)
        @test optData.τ_upper == 0.5 * sqrt(10)
        @test optData.δk == 1.0
        @test params_update_flag
    end 

    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration 
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 0.5
        optData.δk = 1.0 

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)
        
        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 0.5 / sqrt(2)
        @test optData.τ_upper == 0.5 * sqrt(10)
        @test optData.δk == 1.0
        @test params_update_flag
    end

    ############################################################################
    # Case 3: Did satisfy condition + grad-norm larger than upper bound
    ############################################################################
    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration, upper bound on delta is not exceeded during update
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 2.5
        optData.δk = 1.0
        optData.δ_upper = 2.0

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 2.5 / sqrt(2)
        @test optData.τ_upper == 2.5 * sqrt(10)
        @test optData.δk == 1.5
        @test params_update_flag
    end 

    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration, upper bound on delta is exceeded during update
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 2.5
        optData.δk = 1.0
        optData.δ_upper = 1.2

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 2.5 / sqrt(2)
        @test optData.τ_upper == 2.5 * sqrt(10)
        @test optData.δk == 1.2
        @test params_update_flag
    end 
    
    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration, upper bound on delta is not exceeded during update
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 2.5
        optData.δk = 1.0 
        optData.δ_upper = 2.0

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 2.5 / sqrt(2)
        @test optData.τ_upper == 2.5 * sqrt(10)
        @test optData.δk == 1.5
        @test params_update_flag
    end

    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration, upper bound on delta is exceeded during update
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 2.5
        optData.δk = 1.0 
        optData.δ_upper = 1.2

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 2.5 / sqrt(2)
        @test optData.τ_upper == 2.5 * sqrt(10)
        @test optData.δk == 1.2
        @test params_update_flag
    end
    
    ############################################################################
    # Case 4: Did satisfy condition + inside interval
    ############################################################################
    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration, delta upper bound is exceeded
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 1.5
        optData.δk = 1.0
        optData.δ_upper = 1.2

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 1.0
        @test optData.τ_upper == 2.0
        @test optData.δk == 1.2
        @test params_update_flag
    end 

    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration, delta upper bound is not exceeded
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 1.5
        optData.δk = 1.0
        optData.δ_upper = 2.0

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 1.0
        @test optData.τ_upper == 2.0
        @test optData.δk == 1.5
        @test params_update_flag
    end 

    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration, upper bound on delta is exceeded during update
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 1.5
        optData.δk = 1.0 
        optData.δ_upper = 1.2

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 1.0
        @test optData.τ_upper == 2.0
        @test optData.δk == 1.2
        @test params_update_flag
    end
    
    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration, upper bound on delta is not exceeded during update
        xp1_init = rand(dim)
        xp1 = copy(xp1_init)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 1.5
        optData.δk = 1.0 
        optData.δ_upper = 2.0

        ## get constant field values
        constant_field_values = Vector{Any}()
        for symbol in constant_fields
            push!(constant_field_values, getfield(optData, symbol))
        end

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        ## test constant fields
        i = 1
        for symbol in constant_fields
            if typeof(constant_field_values[i]) == String
                @test constant_field_values[i] == getfield(optData, symbol)
            else
                @test constant_field_values[i] ≈ getfield(optData, symbol) 
            end
            i += 1
        end

        @test xp1 != optData.iter_hist[iter]
        @test xp1 == xp1_init
        @test optData.τ_lower == 1.0
        @test optData.τ_upper == 2.0
        @test optData.δk == 1.5
        @test params_update_flag
    end
end