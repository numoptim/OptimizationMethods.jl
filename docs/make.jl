using Documenter, OptimizationMethods

makedocs(
    sitename="OptimizationMethods.jl Documentation",
    pages = [
        "Overview" => "index.md",
        "Problems" => "problems.md",
        "Methods" => "methods.md"
    ]
)