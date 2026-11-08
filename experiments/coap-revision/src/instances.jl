# Problem and starting-point generation for the COAP revision experiments.
# Everything is a deterministic function of the seeds in config/problems.csv and
# config/starts.csv. The package's constructors draw from the global RNG, so we
# call `Random.seed!(seed)` immediately before constructing a problem.

using Random, OptimizationMethods

const OM = OptimizationMethods

"Variance-function labels used throughout (V1..V4)."
const VARIANCE_LABELS = ["V1", "V2", "V3", "V4"]

"Design: n values, m values, replicates per (n, m, V), starts per problem."
const N_VALUES = [10, 50, 90]
const M_VALUES = [100, 1000]
const N_REPLICATES = 5
const N_STARTS = 10

"""
    build_problem(V, n, m, seed) -> NLPModel

Builds the quasi-likelihood problem (logistic link) with variance function `V`
(Appendix C constructor arguments of the old experiment scripts).
"""
function build_problem(V::AbstractString, n::Integer, m::Integer, seed::Integer)
    Random.seed!(seed)
    T = Float64
    if V == "V1"
        return OM.QLLogisticSin(T; nobs = m, nvar = n)
    elseif V == "V2"
        return OM.QLLogisticMonomial(T; nobs = m, nvar = n, p = 2.25, c = 1.0)
    elseif V == "V3"
        return OM.QLLogisticCenteredExp(T; nobs = m, nvar = n, p = 2.25, c = 1.0)
    elseif V == "V4"
        return OM.QLLogisticCenteredLog(T; nobs = m, nvar = n, p = 2.25, c = 1.0, d = 1.0)
    else
        error("unknown variance function $V")
    end
end

"""
    build_start(n, seed) -> Vector{Float64}

Starting point uniform on [-10, 10]^n from an explicit `Xoshiro(seed)` stream.
"""
function build_start(n::Integer, seed::Integer)
    rng = Random.Xoshiro(seed)
    return 20.0 .* rand(rng, n) .- 10.0
end

"""
    true_design(n, m, seed) -> (design, β_true)

Regenerates the design matrix and true coefficients exactly as the constructors
do (same RNG state; `get_design` is the first draw), because the problem structs
do not store `β_true`.
"""
function true_design(n::Integer, m::Integer, seed::Integer)
    Random.seed!(seed)
    return OM.get_design(0.3, Int64(m), Int64(n))
end

"Checksums used by run_job.jl to detect RNG drift across machines/Julia patches."
data_checksums(prob) = (sum(prob.design), sum(abs2, prob.design),
    sum(prob.response), sum(abs2, prob.response))
