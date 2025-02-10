using OptimizationMethods

progData = OptimizationMethods.PoissonRegression(Float64)

subproblem_solver_struct = OptimizationMethods.LipschitzApproxGD(
    Float64, 
    x0=randn(50), 
    init_stepsize=0.0005, 
    threshold=1e-10, 
    max_iterations=100
)
subproblem_solver_function = lipschitz_approximation_gd

optData = ProximalPointGD(Float64;
    x0 = randn(50),
    distance_penalty = 30.0,
    subproblem_solver_struct = subproblem_solver_struct,
    subproblem_solver_function = subproblem_solver_function,
    threshold = 1e-10,
    max_iterations = 1000,
)

x = proximal_point_gd(optData, progData)

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