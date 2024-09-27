# Part of Optimization.jl

"""
    SimpleStats{T}

Mutable struct that keeps track of some simple stats for optimizers.

# Constructors

    SimpleStats(::Type{T})

Constructs a simple stats struct with default values. Everything is set to 0 except
    for `grad_norm` and `status` which are both set to `-1` to indicate the optimization
    did not make a single iteration.

# Fields

- `total_iters :: Int64`, number of total iteration taken by the optimization algorithm
- `grad_norm :: T`, norm of the gradient at the returned point of an optimization algorithm
- `nobj :: Int64`, number of objective evaluations
- `ngrad :: Int64`, number of gradient evaluations
- `nhess :: Int64`, number of hessian evaluations
- `time :: Float64`, elapsed time of the method
- `status :: Tuple{Int64, T}`, First element is status code: `0`` if the `grad_norm` is below some tolerance, or `1`` if the `grad_norm` is above some tolerance, and `-1` upon initialization. Second element is the gradient tolerance checked to compute the status code.
- `status_message :: String`, information about the status of the algorithm.

"""
mutable struct SimpleStats{T}
    total_iters :: Int64
    grad_norm :: T
    nobj :: Int64
    ngrad :: Int64
    nhess :: Int64
    time :: Float64 
    status :: Tuple{Int64, T}
    status_message :: String
end

function SimpleStats(::Type{T}) where T
    return SimpleStats(0, T(-1), 0, 0, 0, 0.0, (-1, T(0)), "")
end