using OptimizationMethods

progData = OptimizationMethods.LogisticRegression(Float64)

################################################################################
# Monotone Version
################################################################################

x0 = randn(50)
optData = OptimizationMethods.NonsequentialArmijoSafeBBGD(Float64,
    x0 = x0,
    init_stepsize = 1e-10,
    long_stepsize = true,
    α_lower = 1e-16,
    α_upper = 1e16,
    δ0 = 1.0,
    δ_upper = 1.0,
    ρ = 1e-4,
    M = 1,
    threshold = 1e-10,
    max_iterations = 1000) 

x = OptimizationMethods.nonsequential_armijo_safe_bb_gd(optData, progData);

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
    Non-monotone: $(length(optData.objective_hist) != 1)
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

################################################################################
# Non-monotone Version
################################################################################

# reset for non-monotone version
progData.counters.neval_obj = 0
progData.counters.neval_grad = 0

optData = OptimizationMethods.NonsequentialArmijoSafeBBGD(Float64,
    x0 = x0,
    init_stepsize = 1e-10,
    long_stepsize = true,
    α_lower = 1e-16,
    α_upper = 1e16,
    δ0 = 1.0,
    δ_upper = 1.0,
    ρ = 1e-4,
    M = 10,
    threshold = 1e-10,
    max_iterations = 1000) 

x = OptimizationMethods.nonsequential_armijo_safe_bb_gd(optData, progData);

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
    Non-monotone: $(length(optData.objective_hist) != 1)
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