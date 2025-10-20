using OptimizationMethods
using QuadGK
using LinearAlgebra
using Random
using Plots
using DataFrames
using CSV

function f(x)
    V(μ) = (abs(μ)^(5/2) + .1)^(-3/2)
    weighted_residual(μ, y) = (y-μ)/V(μ)

    obj = -quadgk(μ -> weighted_residual(μ, 2.), 0, x)[1]
    return obj
end

function h(x)
    t1 = sqrt(abs(x)^(5/2) + .1)
    t2 = (20*abs(x)^(5/2) + (75*x^2 - 150*x)*sqrt(abs(x)) + 2)

    return t1*t2/20
end

# objective
x = range(-1, 2.5, length=100)
y = f.(x)

plot(x, y)

df = DataFrame(:x => x, :y => y)
CSV.write("ql-obj.csv", df)

# hessian
x = range(-1, 2.5, length=100)
y = h.(x)

plot(x, y)

df = DataFrame(:x => x, :y => y)
CSV.write("ql-hess.csv", df)

# (b < a) int_a^b f dx = F(b) - F(a)
# -int_b^a f dx = - (F(a) - F(b)) = F(b) - F(a)

# 0.31328885123138556
#   0.44925940997517455
#  -0.20404514407235996
#  -0.4476036572891464
#  -0.1483709465410491
#  -0.7854597944935658
#  -0.716326700997842
#  -0.8601625263917013
#   ⋮
#  -0.4771218437372792
#   0.7949123366180044
#  -0.5533824225622428
#  -0.3474469627699248
#  -0.8206259551313023
#  -0.8458591964876244