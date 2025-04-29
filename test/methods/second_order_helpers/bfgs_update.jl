# Date: 2025/04/29
# Author: Christian Varner
# Purpose: Test the implementation of the damped BFGS Update

module TestDampedBFGSUpdate

@testset "Test update_bfgs!" begin

    # test when sHs == 0

    # test when sHs > 0 and damped_update = false 

    # test when sHs > 0 and damped_update = true and sy >= .2 sHs

    # test when sHs > 0 and damped_update = true and sy < .2 sHs

    # test sr == 0.0

end # end the test

end # end module