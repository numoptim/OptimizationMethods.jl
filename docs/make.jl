using Documenter
using OptimizationMethods

makedocs(
    sitename = "OptimizationMethods",
    modules = [OptimizationMethods],
    pages = [
        "Home" => "index.md"
        "Table of Contents" => "api/contents.md"
        "API Reference" => [
            "Objective Function Models" => "api/objective_functions.md",
            "Optimization Routines" => "api/optimization_routines.md"
        ]
    ]
    )