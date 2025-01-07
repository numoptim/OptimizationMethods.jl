# Quasi-likelihood Estimation

Quasi-likelihood estimation was first introduced by 
[wedderburn1974Quasilikelihood](@citet) as a way of estimating regression
coefficients when the underlying probability model generating the data is hard to identify, 
or when the data is not explained well by common approaches, such as [generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model). 
We now briefly provide background information on the method, followed by some examples that are implemented in our package.
 
## Quasi-likelihood Setting and Estimation Methodology

Quasi-likelihood estimation, following Wedderburn's description, assumes that the
response variables, $y_i \in \mathbb{R}$, for $i = 1,...,n$, being either discrete or continuous, are independently collected
from a distribution that is only partially known.
Specifically, for covariate vectors, $x_i \in \mathbb{R}^p$, $i = 1,...,n$, and a vector $\theta^{\star} \in \mathbb{R}^p$
the model assumes the following relationship between $y_i, x_i$ and $\theta^{\star}$.

- (Mean Relationship) The expected value of each observation, $\mathbb{E}[y_i | x_i, \theta^{\star}] = \mu_i$, satisfies $\mu_i = \mu(x_i^\intercal \theta^{\star})$ for a known function $\mu : \mathbb{R} \to \mathbb{R}$. Typically, $g$ is selected to be invertible.
- (Variance Relationship) The variance of each data point satisfies $\mathbb{V}[y_i | x_i, \theta^{\star}] = V(\mu(x_i^\intercal \theta^{\star}))$ for a known non-negative function $V : \mathbb{R} \to \mathbb{R}$.

Estimation of $\theta^{\star}$ proceeds by combining these two components to form the quasi-likelihood objective,

$$\min_{\theta} F(\theta) = \min_{\theta} -\sum_{i=1}^n \int_{c_i}^{\mu(x_i^\intercal \theta)} \frac{y_i - m}{V(m)}dm.$$

!!! note
    Since the mean and variance function can be arbitrarily selected, $F(\theta)$ might
    not correspond to any known likelihood function, hence the name quasi-likelihood.
    Furthermore, the objective function $F(\theta)$ might
    not have a closed form expression, requiring numerical integration techniques to evaluate. 
    Owing to the structure of the objective however, the [fundamental theorem
    of calculus](https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus) can be used to compute the gradient.
    This potentially makes the objective more expensive to compute than the gradient, an atypical scenario
    for optimization.

Having presented some background on quasi-likelihood estimation, we now present
a use case of the framework for a special case in semi-parametric regression.

## Example: Semi-parametric Regression with Heteroscedasticity Errors

Suppose that observations, $y_i \in \mathbb{R}$, $i = 1, ..., n$, are independent 
and are associated with covariate vectors $x_i \in \mathbb{R}^p$, $i = 1,...,n$. 
Furthermore, suppose $(x_i, y_i)$ satisfy the following relationship

$$y_i = \mu(x_i^\intercal \theta^{\star}) + V( \mu(x_i^\intercal \theta^{\star}) )^{1/2} \epsilon_i,$$

for a function $\mu:\mathbb{R} \to \mathbb{R}$, a non-negative function 
$V : \mathbb{R} \to \mathbb{R}$, and a vector $\theta^{\star} \in \mathbb{R}^p$.
Here, $\epsilon_i$ are independent realization from a distribution with a
mean and variance of $0$ and $1$, respectively, but whose exact form cannot be fully specified. 

This model is a special form of semi-parametric regression with heteroscedastic errors, also 
satisfying the requirements of the quasi-likelihood estimation framework.
Indeed, the mean and variance relationships for quasi-likelihood are satisfied by checking
the expected value and variance of $y_i$ using the statistical model above.

Below, we provide a list of variance functions that lead to the quasi-likelihood objective being hard
to analytically integrate (if not impossible), some of which appear in literature.

- Let $V : \mathbb{R} \to \mathbb{R}$ be defined as $V(\mu) = 1 + \mu + \sin(2\pi\mu)$. See Section 4 of [lanteri2023Designing](@citet).
- Let $V : \mathbb{R} \to \mathbb{R}$ be defined as $V(\mu) = (\mu^{2})^p + c$ for $c \in \mathbb{R}_{> 0}$ and $p \in \mathbb{R}_{>.5}$. See for example [variance stabilization transformations](https://en.wikipedia.org/wiki/Variance-stabilizing_transformation).
- Let $V : \mathbb{R} \to \mathbb{R}$ be defined as $V(\mu) = \exp(-((\mu - c)^{2})^p)$ for $c \in \mathbb{R}$ and $p \in \mathbb{R}_{>.5}$. See Section 4 of [lanteri2023designing](@citet).
- Let $V : \mathbb{R} \to \mathbb{R}$ be defined as $V(\mu) = \log( ((\mu - c)^{2})^p + 1)$ for $c \in \mathbb{R}$ and $p \in \mathbb{R}_{>.5}$.

## Model Implementation



## References 
```@bibliography
Pages=["quasilikelihood_estimation.md"]
Canonical=false 
```