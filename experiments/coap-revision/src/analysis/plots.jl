# PNG previews (Phase 3.3). Requires CairoMakie (analysis environment).
using CairoMakie

const PALETTE = Makie.wong_colors()
const STYLES = [:solid, :dash, :dot, :dashdot]
style_of(j) = (PALETTE[mod1(j, length(PALETTE))], STYLES[mod1(cld(j, length(PALETTE)), length(STYLES))])

function plot_profiles(curves, set::String, metric::Symbol, path::String)
    fig = Figure(size = (1000, 820))
    local ax1
    for (i, tol) in enumerate(TOLS)
        ax = Axis(fig[div(i - 1, 2) + 1, mod(i - 1, 2) + 1]; xscale = log10,
            xlabel = "τ  (" * METRIC_LABEL[metric] * " ratio to best)", ylabel = "ρ(τ)",
            title = "ε = " * tol_label(tol))
        i == 1 && (ax1 = ax)
        names, taus, ρ = curves[(set, metric, tol)]
        for (j, m) in enumerate(names)
            c, ls = style_of(findfirst(==(m), ALL_METHODS))
            stairs!(ax, taus, ρ[:, j]; label = m, color = c, linestyle = ls, linewidth = 2.5)
        end
        ylims!(ax, 0, 1.02)
    end
    Legend(fig[3, 1:2], ax1; orientation = :horizontal, nbanks = 2, tellheight = true)
    save(path, fig; px_per_unit = 1.5)
end

function plot_reliability(T::DataFrame, bc_by_tol, path::String)
    fig = Figure(size = (1100, 520))
    ax = Axis(fig[1, 1]; xticks = (1:length(ALL_METHODS), ALL_METHODS), xticklabelrotation = π / 4,
        ylabel = "reliability (best configuration)", title = "Reliability with 95% margins")
    cols = Makie.wong_colors()[1:4]
    for (k, tol) in enumerate(TOLS)
        bc = bc_by_tol[tol]
        xs = Float64[]; ys = Float64[]; es = Float64[]
        for r in eachrow(bc)
            rel = Float64.(T.reliable[(T.tol .== tol) .& (T.algorithm_id .== r.algorithm_id)])
            push!(xs, findfirst(==(r.method), ALL_METHODS) + (k - 2.5) * 0.2)
            push!(ys, mean(rel)); push!(es, moe95(rel))
        end
        barplot!(ax, xs, ys; width = 0.18, color = cols[k], label = "ε = " * tol_label(tol))
        errorbars!(ax, xs, ys, es; whiskerwidth = 4, color = :black)
    end
    ylims!(ax, 0, 1.05)
    axislegend(ax; position = :rt, nbanks = 2)
    save(path, fig; px_per_unit = 1.5)
end

function plot_events(ev::DataFrame, path::String, tol::Real)
    fig = Figure(size = (700, 420))
    ax = Axis(fig[1, 1]; xticks = (1:nrow(ev), ev.Events), xticklabelrotation = π / 6,
        ylabel = "average proportion (%)", title = "OURS triggering events, up to ε = " * tol_label(tol))
    y = ev[!, "Average.Proportion.on.Inner.Loops"]
    barplot!(ax, 1:nrow(ev), y; color = PALETTE[1])
    errorbars!(ax, 1:nrow(ev), y, ev[!, "Margin.of.Error"]; whiskerwidth = 6, color = :black)
    save(path, fig; px_per_unit = 1.5)
end
