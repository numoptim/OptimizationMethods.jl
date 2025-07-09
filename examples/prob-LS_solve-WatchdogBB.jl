using OptimizationMethods

progData = OptimizationMethods.LeastSquares(Float64)

x0 = randn(50)
optData = OptimizationMethods.WatchdogSafeBarzilaiBorweinGD(
    Float64;
    x0 = x0,
    init_stepsize = 1e-10,
    long_stepsize = true,
    α_lower = 1e-16,
    α_default = 1e-10,
    δ = .5,
    ρ = 1e-4,
    line_search_max_iterations = 100,
    η = 1e-6,
    inner_loop_max_iterations = 100,
    window_size = 1,
    threshold = 1e-10,
    max_iterations = 100 
)

x = OptimizationMethods.watchdog_safe_barzilai_borwein_gd(optData, progData)

# Compute objective and residual evals during optimization 
obj_evals = progData.counters.neval_obj

# Compute objective values of different iterates for reporting purposes
obj_init = OptimizationMethods.obj(progData, optData.iter_hist[1])
obj_term = OptimizationMethods.obj(progData, 
    optData.iter_hist[optData.stop_iteration + 1])


println(
"""
    Problem: $(progData.meta.name)
    Nonmonotone: $(length(optData.objective_hist) != 1)
    Solver: $(optData.name)
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

# reset for different parameters
progData.counters.neval_obj = 0
progData.counters.neval_grad = 0

optData = OptimizationMethods.WatchdogSafeBarzilaiBorweinGD(
    Float64;
    x0 = x0,
    init_stepsize = 1e-10,
    long_stepsize = true,
    α_lower = 1e-16,
    α_default = 1e-10,
    δ = .5,
    ρ = 1e-4,
    line_search_max_iterations = 100,
    η = 1e-6,
    inner_loop_max_iterations = 100,
    window_size = 3,
    threshold = 1e-10,
    max_iterations = 100 
)

x = OptimizationMethods.watchdog_safe_barzilai_borwein_gd(optData, progData)

# Compute objective and residual evals during optimization 
obj_evals = progData.counters.neval_obj

# Compute objective values of different iterates for reporting purposes
obj_init = OptimizationMethods.obj(progData, optData.iter_hist[1])
obj_term = OptimizationMethods.obj(progData, 
    optData.iter_hist[optData.stop_iteration + 1])


println(
"""
    Problem: $(progData.meta.name)
    Nonmonotone: $(length(optData.objective_hist) != 1)
    Solver: $(optData.name)
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