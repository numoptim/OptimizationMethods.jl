# Date: 02/06/2025
# Author: Christian Varner
# Purpose: Test cases for wolfe ebls procedure

module TestEBLS

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Line Search Helper: EBLS" begin

###############################################################################
# Test definitions
###############################################################################

@test isdefined(OptimizationMethods, :EBLS!)

###############################################################################
# Test functionality implementation 1
###############################################################################



###############################################################################
# Test functionality implementation 2
###############################################################################




end

end