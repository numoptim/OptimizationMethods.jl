# Date: 2025/04/21
# Author: Christian Varner
# Purpose: Test the implementation of (nonmonotone) backtracking 
# with modified newton method

module TestModifiedNewtonStepFixStepSizeNLSMaxVal

using Test, OptimizationMethods, CircularArrays, LinearAlgebra

@testset "Test -- FixedModifiedNewtonNLSMaxValGD Structure" begin

    ############################################################################
    # Test the definition of structure
    ############################################################################
    
    ############################################################################
    # Test the field names in the structure
    ############################################################################

    ############################################################################
    # Test the constructor -- errors
    ############################################################################

    ############################################################################
    # Test the constructor -- field types
    ############################################################################
    
    ############################################################################
    # Test the constructor -- initial values
    ############################################################################
    
end # End structure test cases

@testset "Test -- fixed_modified_newton_nls_maxval_gd (Monotone)" begin
end # end of monotone method implementation test cases

@testset "Test - fixed_modified_newton_nls_maxval_gd (Nonmonotone)" begin
end # end of non-monotone method implementation test cases

end # end of the test cases