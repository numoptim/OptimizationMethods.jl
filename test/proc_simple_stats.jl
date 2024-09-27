# Date: 09/27/2024
# Author: Christian Varner
# Purpose: Test structures defined in Optimization Methods

module ProceduralSimpleStats

using Test, OptimizationMethods, Random

@testset "Simple Stats Struct -- Procedural" begin

    # testing context
    Random.seed!(1010)

    ###############################
    # Test default constructor
    ###############################

    # test for field names
    fn = fieldnames(OptimizationMethods.SimpleStats)
    @test :total_iters in fn
    @test :grad_norm in fn
    @test :nobj in fn
    @test :ngrad in fn
    @test :nhess in fn
    @test :time in fn
    @test :status in fn
    @test :status_message in fn

    # make object
    stats = OptimizationMethods.SimpleStats(Float64)
    @test stats.total_iters == 0
    @test stats.grad_norm == -1
    @test stats.nobj == 0
    @test stats.ngrad == 0
    @test stats.nhess == 0
    @test stats.time == 0.0
    @test stats.status == (-1, 0.0)
    @test stats.status_message == ""

    

end # end of test set

end # end module