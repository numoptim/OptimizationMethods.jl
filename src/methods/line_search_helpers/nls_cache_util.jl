# Date: 01/27/2025
# Author: Christian Varner
# Purpose: Helper functions to maintain the
# maximum value for non-monotone line search

"""
    shift_left!(array::Vector{T}, sz::Int) where {T}

Shift the elements of array one index to the left, where the first element
gets placed in the last position of the array. This function modifies the
vector in place.

!!! note
    The method assumes `array` is non-empty and `sz` correctly corresponds
    to the size of `array`. If this is not the case, we do not ensure proper
    functionality of this method.

# Arguments

- `array::Vector{T}`, array for which elements should be shifted.
- `sz::Int`, size of the array.

# Return

- `nothing`
"""
function shift_left!(array::Vector{T}, sz::Int) where {T}

    if sz > 1
        first_elem = array[1]
        for i in 1:(sz-1)
            array[i] = array[i+1]
        end
        array[sz] = first_elem
    end

    return nothing
end

"""
    update_maximum(array::Vector{T}, previous_max_index::Int64,
        sz::Int) where {T}

Given `array`, find the maximum value given that the elements of `array`
where shifted left and a new element was added at `sz`. The previous maximum 
should have position `previous_max_index` in `array`; however, this could be
an invalid index for `array`.

# Arguments

- `array::Vector{T}`, vector of values of type `T`.
- `previous_max_index::Int64`, index of the maximum value given that
    the elements were shifted left to add a new element at `sz`
- `sz::Int`, size of `array`.

# Return

- `value::T`, value of the maximum of the array.
- `index::Int64`, index of the maximum of the array.
"""
function update_maximum(
    array::Vector{T},
    previous_max_index::Int64,
    sz::Int
    ) where {T}

    # TODO - should we use a max heap here instead?

    # update the maximum value and the index for the maximum value
    # of the array optData.objective_hist
    if sz > 1
        if previous_max_index == 0
            return findmax(array)
        else
            if array[previous_max_index] < array[sz]
                return array[sz], sz
            else
                return array[previous_max_index], previous_max_index
            end
        end
    else
        return array[1], 1
    end
end 