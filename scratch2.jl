using OptimizationMethods
using QuadGK
using LinearAlgebra
using Random
using Plots
using DataFrames
using CSV

μ = 1.
θ = 2.0
y1 = μ .+ randn(100)
y2 = μ/θ .+ randn(100)

function f(x)
    V(θ) = (1 + θ^2)^2
    weighted_residual(θ, yi1, yi2) = (yi2 + θ*yi1)*(yi1-θ*yi2)/V(θ)

    obj = 0
    for i in 1:100
        obj -= quadgk(θ -> weighted_residual(θ, y1[i], y2[i]), 0, x)[1] 
    end
    return obj
end

# objective
x = range(-4, 10, length=100)
y = f.(x)

plot(x, y)

df = DataFrame(:x => x, :y => y)
CSV.write("fc-obj.csv", df)