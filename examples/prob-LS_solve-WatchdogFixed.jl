using OptimizationMethods

progData = OptimizationMethods.LeastSquares(Float64)

x0 = randn(50)
optData = OptimizationMethods.WatchdogFixedGD(
    Float64;
    x0 = x0,
    α = 1.0,
    δ = 0.5,
    ρ = 1e-4,
    line_search_max_iterations = 10,
    window_size = 1,
    η = 1e-6,
    inner_loop_max_iterations = 50,
    threshold = 1e-10,
    max_iterations = 100
)

x = OptimizationMethods.watchdog_fixed_gd(optData, progData)

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

optData = OptimizationMethods.WatchdogFixedGD(
    Float64;
    x0 = x0,
    α = 0.8,
    δ = 0.5,
    ρ = 1e-4,
    line_search_max_iterations = 20,
    window_size = 10,
    η = 1e-6,
    inner_loop_max_iterations = 100,
    threshold = 1e-10,
    max_iterations = 200
)

x = OptimizationMethods.watchdog_fixed_gd(optData, progData)

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