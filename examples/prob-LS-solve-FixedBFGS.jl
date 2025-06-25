using OptimizationMethods

# Define the Least Squares regression problem
progData = OptimizationMethods.LeastSquares(Float64)

# Monotone version
M = 1
x0 = randn(50) 
optData = FixedDampedBFGSNLSMaxValGD(Float64; 
    x0 = randn(50), 
    c = 1e-4, 
    β = 1e-3, 
    α = 1.0,
    δ = .5, 
    ρ = 1e-4, 
    line_search_max_iteration = 100, 
    window_size = M, 
    threshold = 1e-10, 
    max_iterations = 200)

x = fixed_damped_bfgs_nls_maxval_gd(
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

# Reset counters for non-monotone version
progData.counters.neval_obj = 0
progData.counters.neval_grad = 0

# Non-monotone version
M = 10
x0 = randn(50) 
optData = FixedDampedBFGSNLSMaxValGD(Float64; 
    x0 = randn(50), 
    c = 1e-4, 
    β = 1e-3, 
    α = 1.0,
    δ = .5, 
    ρ = 1e-4, 
    line_search_max_iteration = 100, 
    window_size = M, 
    threshold = 1e-10, 
    max_iterations = 200)

x = fixed_damped_bfgs_nls_maxval_gd(
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