# Date: 09/18/2024
# Author: Christian Varner
# Purpose: Implement c/T

"""
"""
function inverse_k(c :: Float64)
    function step_size(k :: Int64)
        return c / k
    end

    return step_size
end