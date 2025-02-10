# Date: 12/19/2024
# Author: Christian Varner
# Purpose: Create an example dataset using the semi-parametric regression
# framework, and try to solve the quasi-likelihood objective function

using OptimizationMethods
using Plots

# create a dataset
qllinearsine = OptimizationMethods.QLLinearSine(
    Float64;
    nobs = 10000,
    nvar = 10)

# display the results in a histogram
p = histogram(qllinearsine.response)
xlabel!(p, "Observation")
ylabel!(p, "Count")
title!(p, "Histogram of Responses with Linear Link and \n Sine Variance with Arcsine Errors")

# create barzilai borwein data 