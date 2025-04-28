# Date: 2025/04/24
# Author: Christian Varner
# Purpose: Illustrate the use of non-sequential bfgs

using OptimizationMethods

progData = OptimizationMethods.QLLogisticSin(Float64)

M = 1
x0 = randn(50) 
inner_loop_radius = 5.0
inner_loop_max_iterations = 10
max_iterations = 500

optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
    x0 = x0,
    c = 1e-4,
    β = 1e-3,
    α = 1.0,
    δ0 = 1.0,
    δ_upper = 1.0,
    ρ = 1e-4,
    M = M,
    inner_loop_radius = inner_loop_radius,
    inner_loop_max_iterations = inner_loop_max_iterations,
    threshold = 1e-10,
    max_iterations = max_iterations)

x = nonsequential_armijo_fixed_damped_bfgs(
    optData,
    progData
)

# Compute objective and residual evals during optimization 
obj_evals = progData.counters.neval_obj

# Compute objective values of different iterates for reporting purposes
obj_init = OptimizationMethods.obj(progData, optData.iter_hist[1])
obj_term = OptimizationMethods.obj(progData, 
    optData.iter_hist[optData.stop_iteration + 1])


println(
"""
    Problem: $(progData.meta.name)
    Solver: $(optData.name)
    Non-monotone: $(M != 1)
    Parameter Dimension: $(progData.meta.nvar)
    
    Max Iterations Allowed: $(optData.max_iterations)
    Gradient Stopping Threshold: $(optData.threshold)

    Initial Objective: $obj_init
    Initial Grad Norm: $(optData.grad_val_hist[1])

    Terminal Iteration: $(optData.stop_iteration)
    Terminal Objective: $obj_term
    Terminal Grad Norm: $(optData.grad_val_hist[optData.stop_iteration + 1])

    Objective Evaluations: $obj_evals
    Gradient Evaluations: $(progData.counters.neval_grad)
    Hessian Evaluations: $(progData.counters.neval_hess)
"""
)

# reset for non-monotone version
progData.counters.neval_obj = 0
progData.counters.neval_grad = 0

M = 10
optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
    x0 = x0,
    c = 1e-4,
    β = 1e-3,
    α = 1.0,
    δ0 = 1.0,
    δ_upper = 1.0,
    ρ = 1e-4,
    M = M,
    inner_loop_radius = inner_loop_radius,
    inner_loop_max_iterations = inner_loop_max_iterations,
    threshold = 1e-10,
    max_iterations = max_iterations)

x = nonsequential_armijo_fixed_damped_bfgs(
    optData,
    progData
)

# Compute objective and residual evals during optimization 
obj_evals = progData.counters.neval_obj

# Compute objective values of different iterates for reporting purposes
obj_init = OptimizationMethods.obj(progData, optData.iter_hist[1])
obj_term = OptimizationMethods.obj(progData, 
    optData.iter_hist[optData.stop_iteration + 1])


println(
"""
    Problem: $(progData.meta.name)
    Solver: $(optData.name)
    Non-monotone: $(M != 1)
    Parameter Dimension: $(progData.meta.nvar)
    
    Max Iterations Allowed: $(optData.max_iterations)
    Gradient Stopping Threshold: $(optData.threshold)

    Initial Objective: $obj_init
    Initial Grad Norm: $(optData.grad_val_hist[1])

    Terminal Iteration: $(optData.stop_iteration)
    Terminal Objective: $obj_term
    Terminal Grad Norm: $(optData.grad_val_hist[optData.stop_iteration + 1])

    Objective Evaluations: $obj_evals
    Gradient Evaluations: $(progData.counters.neval_grad)
    Hessian Evaluations: $(progData.counters.neval_hess)
"""
)