# Dolan–Moré performance profiles (Phase 3.1).
include(joinpath(@__DIR__, "common.jl"))

"""
    profile_ratios(T, config, tol, metric) -> (names, R, n_pairs, n_dropped)

`config` maps method name -> algorithm_id (best configuration). For every
problem-start pair the cost of a solver is its `metric` at the first hit of `tol`
if the run is reliable at `tol`, else `Inf`. For `:nobj` one is added to every
cost (`#F + 1`), because gradient-only methods have `#F = 0` and ratios would be
undefined. `R[p, s] = cost[p, s] / min_s cost[p, s]` over the pairs where at least
one solver succeeds (`n_pairs` of them); the other pairs are dropped (`n_dropped`).
"""
function profile_ratios(T::DataFrame, config::Dict{String,Int}, tol::Real, metric::Symbol)
    names = collect(keys(config))
    sort!(names, by = m -> findfirst(==(m), ALL_METHODS))
    sub = T[T.tol .== tol, :]
    pairs = sort!(unique(zip(sub.problem_id, sub.start_id)) |> collect)
    pidx = Dict(p => i for (i, p) in enumerate(pairs))
    C = fill(Inf, length(pairs), length(names))
    for (j, m) in enumerate(names)
        d = sub[sub.algorithm_id .== config[m], :]
        for row in eachrow(d)
            if row.reliable
                v = row[metric]
                metric == :nobj && (v += 1)
                C[pidx[(row.problem_id, row.start_id)], j] = v
            end
        end
    end
    best = vec(minimum(C, dims = 2))
    keep = isfinite.(best)
    R = C[keep, :] ./ best[keep]
    return names, R, count(keep), count(.!keep)
end

"ρ_s(τ) = fraction of retained pairs with ratio <= τ, for every τ in `taus`."
profile_curves(R::Matrix{Float64}, taus) =
    [size(R, 1) == 0 ? 0.0 : mean(R[:, j] .<= τ) for τ in taus, j in 1:size(R, 2)]

"Geometric τ grid from 1 to the largest finite ratio (at least 2, at most 1e6)."
function tau_grid(R::Matrix{Float64}; npts = 200)
    fin = filter(isfinite, vec(R))
    τmax = isempty(fin) ? 2.0 : clamp(maximum(fin) * 1.05, 2.0, 1e6)
    return exp.(range(0, log(τmax), length = npts))
end

"""
    all_profiles(T; methods_sets) -> (curves::Dict, summary::DataFrame)

For every set ("main", "all"), metric and tolerance computes the profile of the
best configurations. `curves[(set, metric, tol)] = (names, taus, ρ)`.
"""
function all_profiles(T::DataFrame)
    curves = Dict{Tuple{String,Symbol,Float64},Any}()
    summ = NamedTuple[]
    for tol in TOLS
        for (setname, methods) in (("main", MAIN_SET), ("all", ALL_METHODS))
            bc = best_configs(T, tol; methods = methods)
            config = Dict(String(r.method) => Int(r.algorithm_id) for r in eachrow(bc))
            for metric in METRICS
                names, R, n, dropped = profile_ratios(T, config, tol, metric)
                taus = tau_grid(R)
                ρ = profile_curves(R, taus)
                curves[(setname, metric, tol)] = (names, taus, ρ)
                for (j, m) in enumerate(names)
                    push!(summ, (set = setname, metric = String(metric), tol = tol, method = m,
                        algorithm_id = config[m],
                        reliability = bc.reliability[findfirst(==(m), bc.method)],
                        n_pairs_used = n, n_pairs_dropped = dropped,
                        rho_at_1 = n == 0 ? 0.0 : mean(R[:, j] .<= 1.0),
                        rho_final = n == 0 ? 0.0 : mean(isfinite.(R[:, j]))))
                end
            end
        end
    end
    return curves, DataFrame(summ)
end

"Wide table for pgfplots: columns `tau` and one per method."
function curves_to_df(names, taus, ρ)
    df = DataFrame(tau = taus)
    for (j, m) in enumerate(names)
        df[!, Symbol(m)] = ρ[:, j]
    end
    return df
end
