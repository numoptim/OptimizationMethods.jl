# Writes config/problems.csv, config/starts.csv, config/algorithms.csv.
# Run (from experiments/coap-revision): julia --project=. src/make_config.jl
using CSV, DataFrames, Random, Statistics, Printf, OptimizationMethods
include(joinpath(@__DIR__, "instances.jl"))

const CONFIG_DIR = joinpath(@__DIR__, "..", "config")
mkpath(CONFIG_DIR)

# ---------------------------------------------------------------- problems
# seeds: problem_id p gets data seed 1_000_000 + p; (p, s) gets start seed
# 2_000_000 + 100p + s. All distinct and recorded.
data_seed(p) = 1_000_000 + p
start_seed(p, s) = 2_000_000 + 100p + s

rows = NamedTuple[]
pid = 0
for V in VARIANCE_LABELS, n in N_VALUES, m in M_VALUES, r in 1:N_REPLICATES
    global pid += 1
    seed = data_seed(pid)
    prob = build_problem(V, n, m, seed)
    X = prob.design; y = prob.response
    # regenerate θ* and check the paper's design
    Xc, βtrue = true_design(n, m, seed)
    same_design = Xc == X
    μ = OptimizationMethods.logistic.(X * βtrue)
    noise = y .- μ                       # = V(μ)^{1/2} ε
    cs = data_checksums(prob)
    push!(rows, (problem_id = pid, V = V, n = n, m = m, replicate = r, seed = seed,
        y_min = minimum(y), y_max = maximum(y),
        mu_min = minimum(μ), mu_max = maximum(μ),
        abs_noise_max = maximum(abs, noise),
        checks_y_in_01 = all(0 .< y .< 1),
        checks_mu_in_03_07 = all(0.3 .< μ .< 0.7),
        checks_noise_lt_03 = all(abs.(noise) .< 0.3),
        checks_design_regenerated = same_design,
        sum_design = cs[1], sumsq_design = cs[2], sum_response = cs[3], sumsq_response = cs[4]))
end
problems = DataFrame(rows)
CSV.write(joinpath(CONFIG_DIR, "problems.csv"), problems)

# ------------------------------------------------------------------ starts
srows = NamedTuple[]
for p in problems.problem_id, s in 1:N_STARTS
    n = problems.n[p]
    x0 = build_start(n, start_seed(p, s))
    push!(srows, (problem_id = p, start_id = s, seed = start_seed(p, s),
        x0_min = minimum(x0), x0_max = maximum(x0)))
end
starts = DataFrame(srows)
CSV.write(joinpath(CONFIG_DIR, "starts.csv"), starts)

# -------------------------------------------------------------- algorithms
η_grid = [1e-4, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
arows = NamedTuple[]
id = 0
function add!(method, structname, solver, uses_obj; eta = missing, long_stepsize = missing,
        adatr_alpha = missing, astr1_variant = missing, note = "")
    global id += 1
    push!(arows, (algorithm_id = id, method = method, struct_name = structname,
        solver = solver, uses_obj = uses_obj, eta = eta, long_stepsize = long_stepsize,
        adatr_alpha = adatr_alpha, astr1_variant = astr1_variant, note = note))
end
for η in η_grid
    add!("FIXED", "FixedStepGD", "fixed_step_gd", false; eta = η, note = "step_size = eta")
end
for η in η_grid
    add!("DIM", "DiminishingStepGD", "diminishing_step_gd", false; eta = η,
        note = "step_size_function = stepdown_100_step_size, step_size_scaling = eta")
end
for ls in (true, false), η in η_grid
    add!("BBGD", "BarzilaiBorweinGD", "barzilai_borwein_gd", false; eta = η,
        long_stepsize = ls, note = "init_stepsize = eta")
end
for η in η_grid
    add!("LIPAPP", "LipschitzApproxGD", "lipschitz_approximation_gd", false; eta = η,
        note = "init_stepsize = eta")
end
for η in η_grid
    add!("NEST", "NesterovAcceleratedGD", "nesterov_accelerated_gd", false; eta = η,
        note = "step_size = eta")
end
for η in η_grid
    add!("WEIGHT", "WeightedNormDampingGD", "weighted_norm_damping_gd", false; eta = η,
        note = "init_norm_damping_factor = 1/eta")
end
add!("OURS", "NonsequentialArmijoAdaptiveGD", "nonsequential_armijo_adaptive_gd", true;
    note = "delta0 = 1, delta_upper = 1, rho = 1e-4")
add!("ARM", "BacktrackingGD", "backtracking_gd", true;
    note = "alpha = 1, delta = 0.5, rho = 1e-4, line_search_max_iteration = 100")
for η in η_grid
    add!("WATCH", "WatchdogFixedSimpleGD", "watchdog_fixed_simple_gd", true; eta = η,
        note = "alpha = eta, delta = 0.5, rho = 1e-4, line_search_max_iterations = 100, window_size = 1, inner_loop_max_iterations = 100")
end
for ls in (true, false), η in η_grid
    add!("SPEC", "SafeBarzilaiBorweinNLSMaxValGD", "safe_barzilai_borwein_nls_maxval_gd", true;
        eta = η, long_stepsize = ls,
        note = "init_stepsize = eta, delta = 0.5, rho = 1e-4, window_size = 10, line_search_max_iteration = 100, alpha_lower = 1e-10, alpha_default = 1")
end
for α in (0.0, 0.9), η in η_grid
    add!("ADATR", "AdaptiveTrustRegionGD", "adaptive_trust_region_gd", false; eta = η,
        adatr_alpha = α, note = "init_scaling = b0 = 1/eta, alpha, b_min = 1e-4, b_hat_max = ||grad f(x0)||")
end
for v in ("adagrad", "maxg"), η in η_grid
    add!("ASTR1", "ASTR1GD", "astr1_gd", false; eta = η, astr1_variant = v,
        note = "theta = 1/eta, varsigma = 1/100, B_k = 0")
end
algorithms = DataFrame(arows)
CSV.write(joinpath(CONFIG_DIR, "algorithms.csv"), algorithms)

# ----------------------------------------------------------------- summary
println("problems: ", nrow(problems), "  starts: ", nrow(starts), "  algorithms: ", nrow(algorithms))
println("runs: ", nrow(starts) * nrow(algorithms))
for V in VARIANCE_LABELS
    d = problems[problems.V .== V, :]
    @printf("%s  y in (%.3f, %.3f)  mu in (%.3f, %.3f)  max|noise| = %.3f  all y in (0,1): %s  all mu in (0.3,0.7): %s  all |noise|<0.3: %s\n",
        V, minimum(d.y_min), maximum(d.y_max), minimum(d.mu_min), maximum(d.mu_max),
        maximum(d.abs_noise_max), all(d.checks_y_in_01), all(d.checks_mu_in_03_07), all(d.checks_noise_lt_03))
end
println("design regenerated exactly for all problems: ", all(problems.checks_design_regenerated))
