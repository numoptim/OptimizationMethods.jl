# Contents

```@contents
Pages=["api_problems.md"]
```

# Regression Problems

## Generalized Linear Models

```@docs
OptimizationMethods.LeastSquares

OptimizationMethods.LogisticRegression

OptimizationMethods.PoissonRegression
```

## Quasi-likelihood Objectives

```@docs
OptimizationMethods.QLLogisticSin

OptimizationMethods.QLLogisticCenteredExp

OptimizationMethods.QLLogisticMonomial
```

# Problem Utility

## Regression Link Functions
```@docs
OptimizationMethods.logistic

OptimizationMethods.inverse_complimentary_log_log

OptimizationMethods.inverse_probit
```

## Regression Link Function Derivatives

```@docs
OptimizationMethods.dlogistic

OptimizationMethods.ddlogistic
```

## Regression Variance Functions
```@docs
OptimizationMethods.monomial_plus_constant

OptimizationMethods.linear_plus_sin

OptimizationMethods.centered_exp

OptimizationMethods.centered_shifted_log
```

## Regression Variance Function Derivatives

```@docs
OptimizationMethods.dlinear_plus_sin

OptimizationMethods.dcentered_exp

OptimizationMethods.dmonomial_plus_constant
```

# Index

```@index
```