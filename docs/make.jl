using Documenter
using OptimizationMethods

makedocs(
    sitename = "OptimizationMethods",
    modules = [OptimizationMethods]
    )

deploydocs(
    repo = "github.com/numoptim/OptimizationMethods.jl"
)