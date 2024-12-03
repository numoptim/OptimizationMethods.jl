push!(LOAD_PATH, "../src/")

using Documenter, OptimizationMethods

makedocs(
    sitename="OptimizationMethods.jl Documentation",
    pages = [
        "Overview" => "index.md",
        "Manual" =>[
            "Problems" => [
                "Quasi-likelihood Estimation" => "./problems/quasilikelihood_estimation.md"
            ],
        ],
        "API" => [
            "Problems" => "api_problems.md",
            "Methods" => "api_methods.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/numoptim/OptimizationMethods.jl",
)
