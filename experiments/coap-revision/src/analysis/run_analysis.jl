# End-to-end analysis (Gate 3). From experiments/coap-revision:
#   julia --project=analysis src/analysis/run_analysis.jl [--merged DIR] [--out DIR] [--no-plots]
# Defaults: --merged results/merged, --out results/paper.
include(joinpath(@__DIR__, "profiles.jl"))
include(joinpath(@__DIR__, "tables.jl"))
const PLOTS = !("--no-plots" in ARGS)
PLOTS && include(joinpath(@__DIR__, "plots.jl"))

const ROOT = joinpath(@__DIR__, "..", "..")

function main(args)
    opt(name, default) = (i = findfirst(==(name), args); i === nothing ? default : args[i+1])
    merged = opt("--merged", joinpath(ROOT, "results", "merged"))
    out = opt("--out", joinpath(ROOT, "results", "paper"))
    mkpath(out)
    runs, hits, events = load_merged(merged)
    algs = CSV.read(joinpath(ROOT, "config", "algorithms.csv"), DataFrame)
    T = build_table(runs, hits)
    files = Pair{String,String}[]          # filename => description (for the README)
    write_csv(name, df, desc) = (CSV.write(joinpath(out, name), df); push!(files, name => desc))

    bc_by_tol = Dict(tol => best_configs(T, tol) for tol in TOLS)
    bcdf = vcat([insertcols(bc_by_tol[tol], 1, :tol => tol) for tol in TOLS]...)
    write_csv("best_configurations.csv", bcdf, "best configuration (highest reliability, ties by median #∇F) of every method at every tolerance")

    # reliability and performance tables (Appendix D format)
    for tol in TOLS, metric in METRICS
        write_csv("reliability_$(tol_label(tol))_$(metric).csv",
            reliability_table(T, bc_by_tol[tol], tol, metric),
            "reliability (best configurations) and mean $(METRIC_LABEL[metric]) over reliable runs, ε = $(tol_label(tol))")
    end
    write_csv("all_algorithms_reliability.csv", all_algorithms_table(T, algs),
        "all 100 algorithms with hyperparameters and reliability at each tolerance (supplement algorithm table)")
    write_csv("success_summary.csv", success_summary(T, bc_by_tol),
        "mean and median #F, #∇F, time and weighted cost of reliable runs, best configurations")

    # OURS triggering events (Table 4)
    ev_by_tol = Dict{Float64,DataFrame}()
    for tol in TOLS
        ev_by_tol[tol] = events_table(events, T, tol)
        write_csv("events_$(tol_label(tol)).csv", ev_by_tol[tol],
            "OURS triggering-event proportions (percent) up to the ε = $(tol_label(tol)) hit, and rejection rate")
    end

    # performance profiles
    curves, summ = all_profiles(T)
    for ((set, metric, tol), (names, taus, ρ)) in sort(collect(curves), by = x -> (x[1][1], string(x[1][2]), x[1][3]))
        write_csv("profile_$(set)_$(metric)_$(tol_label(tol)).csv", curves_to_df(names, taus, ρ),
            "performance profile ρ(τ), $(set == "main" ? "main set" : "all methods"), $(METRIC_LABEL[metric]), ε = $(tol_label(tol))")
    end
    write_csv("profile_summary.csv", summ,
        "per set/metric/tolerance: configuration used, reliability, ρ(1), ρ(∞), pairs used and dropped")

    # algorithm table for the supplement
    write_csv("algorithms_table.csv", algs, "config/algorithms.csv: the 100 algorithms and their parameters (supplement algorithm table)")

    if PLOTS
        for set in ("main", "all"), metric in METRICS
            f = "profile_$(set)_$(metric).png"
            plot_profiles(curves, set, metric, joinpath(out, f))
            push!(files, f => "PNG preview of the four profiles (one panel per ε): $(set == "main" ? "main set" : "all methods"), $(METRIC_LABEL[metric])")
        end
        plot_reliability(T, bc_by_tol, joinpath(out, "reliability.png"))
        push!(files, "reliability.png" => "PNG preview: reliability with 95% margins, best configurations")
        plot_events(ev_by_tol[1e-3], joinpath(out, "events_1e-3.png"), 1e-3)
        push!(files, "events_1e-3.png" => "PNG preview: OURS triggering events up to the 1e-3 hit")
    end
    write_readme(out, files, merged, nrow(runs))
    println("analysis done: $(length(files)) files in $out")
end

function write_readme(out, files, merged, nruns)
    mapping = [
        "profile_main_*.csv, profile_main_*.png" => "**Fig. 3 and its discussion** (replaced): performance profiles of the main set (OURS, ARM, WATCH, SPEC, ADATR, ASTR1, LIPAPP) on #F, #∇F, weighted cost and wall time, one per tolerance.",
        "profile_all_*.csv, profile_all_*.png" => "Appendix version of the profiles: all 12 methods.",
        "reliability_<ε>_<metric>.csv, reliability.png" => "**Reliability and performance tables**: one row per method (best configuration), per tolerance, Appendix-D columns, metric averaged over reliable runs.",
        "events_<ε>.csv, events_1e-3.png" => "**Table 4 (triggering events)**: OURS event proportions and rejection rate.",
        "algorithms_table.csv, all_algorithms_reliability.csv" => "**Supplement algorithm table** (the 100 algorithms).",
        "best_configurations.csv, profile_summary.csv, success_summary.csv" => "supporting numbers (which configuration is 'best', ρ(1) and ρ(∞), mean/median costs of reliable runs).",
    ]
    open(joinpath(out, "README.md"), "w") do io
        println(io, "# results/paper: analysis outputs\n")
        println(io, "Generated by `julia --project=analysis src/analysis/run_analysis.jl` from `$(relpath(merged, dirname(out)))` ($(nruns) runs). Do not edit by hand.\n")
        println(io, "All CSVs are plain comma-separated with a header row, ready for `\\addplot table[col sep=comma]` (pgfplots). PNGs are previews only.\n")
        println(io, "## Mapping to the manuscript\n")
        for (f, d) in mapping
            println(io, "* `", f, "`: ", d)
        end
        println(io, "\n## Conventions\n")
        println(io, "* **Reliable at ε**: the method hit ε and F(x_hit) < F(x_0) (the paper's definition, per tolerance).")
        println(io, "* **Best configuration** of a method at ε: highest reliability at ε over its hyperparameter grid; ties are broken by the smaller median #∇F of the reliable runs.")
        println(io, "* **Performance profiles** (Dolan–Moré): cost of a solver on a problem-start pair is its metric at the first hit of ε if the run is reliable, else ∞. ρ_s(τ) is the fraction of pairs with ratio to the best solver ≤ τ. Pairs that no solver of the set solves are dropped (`n_pairs_dropped` in `profile_summary.csv`). For #F the metric is #F + 1 (gradient-only methods have #F = 0).")
        println(io, "* **Weighted cost** = #F·c + #∇F with c = (mean time per F)/(mean time per ∇F) computed per run from that run's timings (c = 0 for methods without F evaluations).")
        println(io, "* **Margins** are 95% normal-approximation margins 1.96·sd/√n (0 when n < 2). In the Appendix-D tables `total.x` = problem-start pairs, `total.y` = reliable runs.")
        println(io, "* **Events** are percentages of inner loops, computed per run up to the outer iteration of the first ε hit and averaged over runs; flags are not mutually exclusive.")
        println(io, "\n## Files\n")
        for (f, d) in sort(files, by = first)
            println(io, "* `", f, "`: ", d)
        end
    end
end

main(ARGS)
