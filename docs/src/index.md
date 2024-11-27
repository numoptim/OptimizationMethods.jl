# Overview

[OptimizationMethods](https://github.com/numoptim/OptimizationMethods.jl) is a 
Julia library for implementing and comparing optimization methods with a focus
on problems arising in data science.
The library is primarily designed to serve those researching optimization 
methods for data science applications.
Accordingly, the library is not implementing highly efficient versions of
these methods, even though we do our best to make preliminary efficiency
optimizations to the code.

There are two primary components to this library.

- **Problems**, which are implementations of optimization problems primarily
    arising in data science. At the moment, problems follow the guidelines
    provided by 
    [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl). 

- **Methods**, which are implementations of important optimization methods
    that appear in the literature. 


The library is still in its infancy and will continue to evolve rapidly.
To understand how to use the library, we recommend looking in the examples
directory to see how different problems are instantiated and how optimization
methods can be applied to them.
We also recommend looking at the docstring for specific problems and methods
for additional details.

# Manual

The manual section includes descriptions of problems and methods that require
a bit more explanation than what is appropriate for in a docstring.

# API

The API section contains explanations for all problems and methods available in 
the library. This is a super set of what is contained in the manual. 

