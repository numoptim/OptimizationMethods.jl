# Date: 09/25/2025
# Author: Christian Varner
# Purpose: Run tests

using Test

@testset verbose=true "OptimizationMethods.jl" begin
    for file in readlines(joinpath(@__DIR__, "test.txt"))
        include(file)
    end
end