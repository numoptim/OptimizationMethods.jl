# Date: 09/18/2024
# Author: Christian Varner
# Purpose: Implement c/sqrt{k} step size

"""
TODO - documentation
"""
function root_k(c :: Float64)
    function step_size(k :: Int64)
        return c / sqrt(k)
    end
    
    return step_size
end