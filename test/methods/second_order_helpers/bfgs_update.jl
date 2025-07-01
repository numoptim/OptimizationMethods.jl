# Date: 2025/04/29
# Author: Christian Varner
# Purpose: Test the implementation of the damped BFGS Update

module TestDampedBFGSUpdate

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Test update_bfgs!" begin

    ###########################################################################
    # test when sHs == 0
    ###########################################################################
    dim = 50
    H = zeros(dim, dim)
    r = zeros(dim)
    update = zeros(dim, dim)
    s = zeros(dim)
    y = zeros(dim)
    let H = H, r = r, update = update, s = s, y = y

        # try to update BFGS approximation
        r_copy = copy(r)
        update_copy = copy(update)
        H_copy = copy(H)
        success = OptimizationMethods.update_bfgs!(
            H, r, update, s, y; damped_update = false
        )

        # update should fail and everything should remain the same
        @test !success
        @test update == update_copy
        @test r == r_copy
        @test H == H_copy

        # try to update BFGS approximation
        r_copy = copy(r)
        update_copy = copy(update)
        H_copy = copy(H)
        success = OptimizationMethods.update_bfgs!(
            H, r, update, s, y; damped_update = true
        )

        # update should fail and everything should remain the same
        @test !success
        @test update == update_copy
        @test r == r_copy
        @test H == H_copy
    
    end

    ###########################################################################
    # test when sHs > 0 and damped_update = false 
    ###########################################################################
    
    Random.seed!(1010)
    dim = 50
    H_half = randn(dim, dim)
    H = H_half' * H_half
    r = randn(dim)
    update = randn(dim, dim)
    s = randn(dim)
    y = H * s
    
    let H = H, r = r, update = update, s = s, y = y
        
        H_copy = copy(H)
        success = OptimizationMethods.update_bfgs!(H, r, update, s, y;
            damped_update = false)

        # test success, update, and r
        @test success
        @test H - update ≈ H_copy
        @test r == y

        # test that update is correctly formed
        Hs = H_copy * s
        BFGS_UPDATE = - Hs * transpose(Hs) ./ dot(s, Hs) + 
            r*transpose(r) ./ dot(s, r)
        @test update ≈ BFGS_UPDATE

    end

    ###########################################################################
    # test when sHs > 0 and damped_update = true and sy >= .2 sHs
    ###########################################################################

    dim = 50
    H_half = randn(dim, dim)
    H = H_half' * H_half
    r = randn(dim)
    update = randn(dim, dim)
    s = randn(dim)
    y = .2 * H * s

    let H = H, r = r, update = update, s = s, y = y
        
        H_copy = copy(H)
        success = OptimizationMethods.update_bfgs!(H, r, update, s, y;
            damped_update = true)

        # test success, update, and r
        @test success
        @test H - update ≈ H_copy
        @test r ≈ y

        # test that update is correctly formed
        Hs = H_copy * s
        BFGS_UPDATE = - Hs * transpose(Hs) ./ dot(s, Hs) + 
            r*transpose(r) ./ dot(s, r)
        @test update ≈ BFGS_UPDATE

    end 


    ###########################################################################
    # test when sHs > 0 and damped_update = true and sy < .2 sHs
    ###########################################################################
    
    dim = 50
    H_half = randn(dim, dim)
    H = H_half' * H_half
    r = randn(dim)
    update = randn(dim, dim)
    s = randn(dim)
    y = .19 * H * s

    let H = H, r = r, update = update, s = s, y = y
        
        H_copy = copy(H)
        success = OptimizationMethods.update_bfgs!(H, r, update, s, y;
            damped_update = true)

        # test success and update
        @test success
        @test H - update ≈ H_copy

        # correct r value
        Hcs = H_copy * s
        sHcs = dot(s, Hcs) 
        sy = dot(s, y)
        θ = (.8 * sHcs) / (sHcs - sy)
        r_correct = θ .* y .+ (1 - θ) .* Hcs
        @test r == r_correct

        # test that update is correctly formed
        BFGS_UPDATE = - Hcs * transpose(Hcs) ./ dot(s, Hcs) + 
            r*transpose(r) ./ dot(s, r)
        @test update ≈ BFGS_UPDATE

    end 

    ###########################################################################
    # test sr == 0.0 and damped_update = false
    ###########################################################################

    dim = 50
    H_half = randn(dim, dim)
    H = H_half' * H_half
    r = randn(dim)
    update = randn(dim, dim)
    s = randn(dim)
    y = zeros(dim)
    
    let H = H, r = r, update = update, s = s, y = y

        H_copy = copy(H)
        success = OptimizationMethods.update_bfgs!(H, r, update, s, y;
            damped_update = false)

        # r = y => dot(s, r) == 0.0
        @test !success
        @test r ≈ y

    end
    
    ###########################################################################
    # test sr == 0.0 and damped_update = true
    ###########################################################################

    dim = 50
    H = -Matrix{Float64}(I, dim, dim)
    r = randn(dim)
    update = randn(dim, dim)
    s = zeros(dim)
    s[1] = 1.0

    y = zeros(dim)
    
    let H = H, r = r, update = update, s = s, y = y

        H_copy = copy(H)
        success = OptimizationMethods.update_bfgs!(H, r, update, s, y;
            damped_update = true)

        # r = y => dot(s, r) == 0.0
        @test !success
        @test r ≈ y

    end

    ###########################################################################
    # test sr == 0.0 and damped_update = true
    ###########################################################################
    dim = 50
    H = -Matrix{Float64}(I, dim, dim)
    r = randn(dim)
    update = randn(dim, dim)
    s = zeros(dim)
    s[1] = 1.0

    y = (.2 + 1e-16) * H * s
    
    let H = H, r = r, update = update, s = s, y = y

        H_copy = copy(H)
        success = OptimizationMethods.update_bfgs!(H, r, update, s, y;
            damped_update = true)

        # -.8/(- 1 - -.2) = -.8/(.4) = -.5
        # r = y => dot(s, r) == 0.0
        @test !success
        @test r ≈ y
    end

end # end the test

end # end module