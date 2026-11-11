module MWE
using LinearAlgebra
using OptimizationModels: OptimizationProblem
import OptimizationProblems: grad!, obj!
export AbstractMethod, History, StoppingCriteria, allocate!, allocate_common!, initialize!, step!, optimize!, check_stop, record!
export FixedStepGD, BacktrackingGD, NesterovAcceleratedGD
include("core.jl")
include("fixed_step_gd.jl")
include("line_search.jl")
include("backtracking_gd.jl")
include("nesterov_accelerated_gd.jl")
end
