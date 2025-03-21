# Date: 2025/21/03
# Author: Christian Varner
# Purpose: Test the non-monotone line search method
# with fixed step size and negative gradient directions

module TestFixedStepNLSMaxValGD{T} <: AbstractOptimizerData{T}

@testset "Test FixedStepNLSMaxValGD{T} -- Structure" begin

    ############################################################################
    # Test structure definition
    ############################################################################

    # test that the structure is defined

    # test optimizer agnostic fields are present

    # test optimizer specific fields are present

    ############################################################################
    # Test the constructor
    ############################################################################

    # test the field types
    field_types(type::T) where {T} =
        [(:name, String),
        (:α, type),
        (:δ, type),
        (:ρ, type),
        (:window_size, Int64),
        (:line_search_max_iteration, Int64),
        (:objective_hist, Vector{type}),
        (:max_value, type),
        (:threshold, type),
        (:max_iterations, Int64),
        (:iter_hist, Vector{Vector{type}}),
        (:grad_val_hist, Vector{type}),
        (:stop_iteration, Int64)]

    let field_types = field_types
        
        # sample random field values

        # test the field types returned by outer constructor
    
    end

    # test that the outer constructor sets the values of field correctly
    let field_types = field_types
        
        #
    
    end

end # end test set for structure

@testset "Test FixedStepNLSMaxValGD{T} -- Method" begin

end # end test for for method

end # End module