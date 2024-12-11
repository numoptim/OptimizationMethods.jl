using Documenter, DocumenterCitations
using OptimizationMethods

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "optimizationmethodsjl.bib");
    style=:authoryear
)

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
        ],
        "References" => "references.md",
    ];
    plugins=[bib]
)

deploydocs(
    repo = "github.com/numoptim/OptimizationMethods.jl",
)
