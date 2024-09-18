# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Implement lipschitz approximation

"""
TODO - Documentation
"""
function lipschitz_approximation_gd(
    func :: AbstractNLPModel{T, S},
    x0 :: S,
    max_iter :: Int64;
    alfa0 :: T = 1e-4
) where S <: Vector{T} where T <: Real

    # initialize iterate
    xprev = zeros(T, size(x0))
    xk = zeros(T, size(x0))
    xprev .= x0
    xk .= x0

    # initialize buffer
    gprev = zeros(T, size(x0))
    gk = zeros(T, size(x0))

    # previous step size
    alfa_prev :: T = zero(T)
    alfak :: T = zero(T)
    wk :: T = zero(T)

    # first iteration
    grad!(gk, func, x0)
    xk .-= alfa0 .* gk
    gprev .= gk
    alfa_prev = alfa0

    # main loop
    k = 2
    while k <= max_iter
        # compute gradient
        grad!(gk, func, xk)

        # efficient
        xprev .*= -1
        xprev .+= xk
        
        gprev .*= -1
        gprev .+= gk
        
        # compute step size
        if k == 2
            alfak = norm(xprev) / (2 * norm(gprev))
        else
            alfak = min(sqrt(1+wk) * alfa_prev, norm(xprev) / (2 * norm(gprev))) 
        end

        # update iterate and weights
        xk .-= alfak .* gk
        wk = alfak / alfa_prev
        alfa_prev = alfak
        
        # update counter
        k += 1
    end

    # return main iterate
    return xk
end