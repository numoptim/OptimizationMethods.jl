var documenterSearchIndex = {"docs":
[{"location":"api_problems/#Contents","page":"Problems","title":"Contents","text":"","category":"section"},{"location":"api_problems/","page":"Problems","title":"Problems","text":"Pages=[\"api_problems.md\"]","category":"page"},{"location":"api_problems/#Regression-Problems","page":"Problems","title":"Regression Problems","text":"","category":"section"},{"location":"api_problems/","page":"Problems","title":"Problems","text":"OptimizationMethods.LeastSquares\n\nOptimizationMethods.LogisticRegression\n\nOptimizationMethods.PoissonRegression","category":"page"},{"location":"api_problems/#OptimizationMethods.LeastSquares","page":"Problems","title":"OptimizationMethods.LeastSquares","text":"LeastSquares{T,S} <: AbstractNLSModel{T,S}\n\nImplements the data structure for defining a least squares problem.\n\nObjective Function\n\nmin_x 05Vert F(x) Vert_2^2\n\nwhere\n\nF(x) = A * x - b\n\nA is the coefficient matrix. b is the constant vector.\n\nFields\n\nmeta::NLPModelMeta{T, S}, data structure for nonlinear programming models\nnls_meta::NLSMeta{T, S}, data structure for nonlinear least squares models\ncounters::NLSCounters, counters for nonlinear least squares models\ncoef::Matrix{T}, coefficient matrix, A, for least squares problem \ncons::Vector{T}, constant vector, b, for least squares problem\n\nConstructors\n\nLeastSquares(::Type{T}; nequ=1000, nvar=50) where {T}\n\nConstructs a least squares problems with 1000 equations and 50 unknowns,     where the entries of the matrix and constant vector are independent     standard Gaussian variables.\n\nArguments\n\nT::DataType, specific data type of the optimization parameter\n\nOptional Keyword Arguments\n\nnequ::Int64=1000, the number of equations in the system \nnvar::Int64=50, the number of parameters in the system \nLeastSquares(design::Matrix{T}, response::Vector{T};        x0 = ones(T, size(design, 2))) where {T}    \n\nConstructs a least squares problem using the design as the coef matrix and the response as the cons vector. \n\nArguments\n\ndesign::Matrix{T}, coefficient matrix for least squares.\nresponse::Vector{T}, constant vector for least squares.\n\nOptional Keyword Arguments\n\nx0::Vector{T}=ones(T, size(design, 2)), default starting point for    optimization algorithms.\n\n\n\n\n\n","category":"type"},{"location":"api_problems/#OptimizationMethods.LogisticRegression","page":"Problems","title":"OptimizationMethods.LogisticRegression","text":"LogisticRegression{T,S} <: AbstractNLPModel{T,S}\n\nImplements logistic regression problem with canonical link function. If     the covariate (i.e., design) matrix and response vector are not supplied,     then these are simulated. \n\nObjective Function\n\nLet A be the covariate matrix and b denote the response vector. The     rows of A and corresponding entry of b correspond to      the predictors and response for the same experimental unit.      Let A_i be the vector that is row i of A,     and let b_i be the i entry of b. Note, b_i is either 0 or      1.\n\nLet mu(x) denote a vector-valued function whose ith entry is \n\nmu_i(x) = frac11 + exp(-A_i^intercal x)\n\nIf A has n rows (i.e., n is the number of observations),  then the objective function is negative log-likelihood function given by\n\nF(x) = -sum_i=1^n b_i log( mu_i(x) ) + (1 - b_i) log(1 - mu_i(x))\n\nFields\n\nmeta::NLPModelMeta{T,S}, data structure for nonlinear programming models\ncounters::Counters, counters for a nonlinear programming model\ndesign::Matrix{T}, the design matrix, A, of the logistic regression   problem\nresponse::Vector{Bool}, the response vector, b, of the logistic   regression problem\n\nConstructors\n\nLogisticRegression(::Type{T}; nobs::Int64 = 1000, nvar::Int64 = 50) where T\n\nConstructs a simulated problem where the number of observations is nobs and     the dimension of the parameter is nvar. The generated design matrix's     first column is all 1s. The remaining columns are independent random     normal entries such that each row (excluding the first entry) has unit     variance. The design matrix is stored as type Matrix{T}.\n\nLogisticRegression(design::Matrix{T}, response::Vector{Bool};\n    x0::Vector{T} = ones(T, size(design, 2)) ./ sqrt(size(design, 2))\n    ) where T\n\nConstructs a LogisticRegression problem with design matrix design and      response vector response. The default initial iterate, x0 is      a scaling of the vector of ones. x0 is optional. \n\n\n\n\n\n","category":"type"},{"location":"api_problems/#OptimizationMethods.PoissonRegression","page":"Problems","title":"OptimizationMethods.PoissonRegression","text":"PoissonRegression{T, S} <: AbstractNLPModel{T, S}\n\nImplements Poisson regression with the canonical link function. If the design     matrix (i.e., the covariates) and responses are not supplied, they are      randomly generated. \n\nObjective Function\n\nLet A be the design matrix, and b be the responses. Each row of A     and corresponding entry in b are the predictor and observation from one      unit. The entries in b must be integer valued and non-negative.\n\nLet A_i be row i of A and b_i entry i of b. Let\n\nmu_i(x) = exp(A_i^intercal x)\n\nLet n be the number of rows of A (i.e., number of observations), then     the negative log-likelihood of the model is \n\nsum_i=1^n mu_i(x) - b_i (A_i^intercal x) + C(b)\n\nwhere C(b) is a constant depending on the data. We implement the objective       function to be the negative log-likelihood up to the constant term C(b).      That is,\n\nF(x) = sum_i=1^n mu_i(x) - b_i (A_i^intercal x)\n\nwarn: Warn\nBecause the additive term C(b) is not included in the objective function, the objective function can take on negative values.\n\nFields\n\nmeta::NLPModelMeta{T, S}, NLPModel struct for storing meta information for    the problem\ncounters::Counters, NLPModel Counter struct that provides evaluations    tracking.\ndesign::Matrix{T}, covariate matrix for the problem/experiment (A).\nresponse::Vector{T}, observations for the problem/experiment (b).\n\nConstructors\n\nPoissonRegression(::Type{T}; nobs::Int64 = 1000, nvar::Int64 = 50) where {T}\n\nConstruct the struct for Poisson Regression when simulated data is needed.      The design matrix (A) and response vector b are randomly generated      as follows.      For the design matrix, the first column is all ones, and the rest are     generated according to a normal distribution where each row has been      scaled to have unit variance (excluding the first column).      For the response vector, let beta be the \"true\" relationship between      the covariates and response vector for the poisson regression model,      then the ith entry of the response vector is generated from a Poisson      Distribution with rate parameter exp(A_i^intercal beta).\n\nPoissonRegression(design::Matrix{T}, response::Vector{T}; \n    x0::Vector{T} = zeros(T, size(design)[2])) where {T}\n\nConstructs the struct for Poisson Regression when the design matrix and response      vector are known. The initial guess, x0 is a keyword argument that is set      to all zeros by default. \n\n!!! Remark     When using this constructor, the number of rows of design must be equal to      the size of response. When providing x0, the number of entries must be the      same as the number of columns in design.\n\n\n\n\n\n","category":"type"},{"location":"api_problems/#Index","page":"Problems","title":"Index","text":"","category":"section"},{"location":"api_problems/","page":"Problems","title":"Problems","text":"","category":"page"},{"location":"references/#References","page":"References","title":"References","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"Barzilai, J. and Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. IMA Journal of Numerical Analysis 8.\n\n\n\nLanteri, A.; Leorato, S.; Lopez-Fidalgo, J. and Tommasi, C. (2023). Designing to detect heteroscedasticity in a regression model. Journal of the Royal Statistical Society 85, 315–326.\n\n\n\nMalitsky, Y. and Mishchenko, K. (2020-11-21). Adaptive Gradient Descent without Descent. In: Proceedings of the 37th International Conference on Machine Learning (PMLR); pp. 6702–6712.\n\n\n\nWedderburn, R. W. (1974). Quasi-Likelihood Functions, Generalized Linear Models, and the Gauss—Newton Method. Biometrika 61, 439–447.\n\n\n\n","category":"page"},{"location":"api_methods/#Contents","page":"Methods","title":"Contents","text":"","category":"section"},{"location":"api_methods/","page":"Methods","title":"Methods","text":"Pages=[\"api_methods.md\"]","category":"page"},{"location":"api_methods/#Barzilai-Borwein-Method","page":"Methods","title":"Barzilai Borwein Method","text":"","category":"section"},{"location":"api_methods/","page":"Methods","title":"Methods","text":"barzilai_borwein_gd\nBarzilaiBorweinGD","category":"page"},{"location":"api_methods/#OptimizationMethods.barzilai_borwein_gd","page":"Methods","title":"OptimizationMethods.barzilai_borwein_gd","text":"barzilai_borwein_gd(optData::BarzilaiBorweinGD{T}, progData::P \n    where P <: AbstractNLPModel{T, S}) where {T,S}\n\nImplements gradient descent with Barzilai-Borwein step size and applies the      method to the optimization problem specified by progData. \n\nReference(s)\n\nBarzilai and Borwein. \"Two-Point Step Size Gradient Methods\". IMA Journal of      Numerical Analysis.\n\nMethod\n\nGiven iterates lbrace x_0ldotsx_krbrace, the iterate x_k+1     is equal to x_k - alpha_k nabla f(x_k), where alpha_k is     one of two versions.\n\nLong Step Size Version (if optData.long_stepsize==true)\n\nIf k=0, then alpha_0 is set to optData.init_stepsize. For k0,\n\nalpha_k = frac Vert x_k - x_k-1 Vert_2^2(x_k - x_k-1^intercal \n    (nabla f(x_k) - nabla f(x_k-1)))\n\nShort Step Size Version (if optData.long_stepsize==false)\n\nIf k=0, then alpha_0 is set to optData.init_stepsize. For k0,\n\nalpha_k = frac(x_k - x_k-1^intercal (nabla f(x_k) - \n    nabla f(x_k-1)))Vert nabla f(x_k) - nabla f(x_k-1)Vert_2^2\n\nArguments\n\noptData::BarzilaiBorweinGD{T}, the specification for the optimization method.\nprogData<:AbstractNLPModel{T,S}, the specification for the optimization   problem. \n\nwarning: Warning\nprogData must have an initialize function that returns subtypes of AbstractPrecompute and AbstractProblemAllocate, where the latter has a grad argument.\n\n\n\n\n\n","category":"function"},{"location":"api_methods/#OptimizationMethods.BarzilaiBorweinGD","page":"Methods","title":"OptimizationMethods.BarzilaiBorweinGD","text":"BarzilaiBorweinGD{T} <: AbstractOptimizerData{T}\n\nA structure for storing data about gradient descent using the Barzilai-Borwein      step size, and the progress of its application on an optimization problem.\n\nFields\n\nname:String, name of the solver for reference.\ninit_stepsize::T, initial step size to start the method. \nlong_stepsize::Bool, flag for step size; if true, use the long version of    Barzilai-Borwein. If false, use the short version. \nthreshold::T, gradient threshold. If the norm gradient is below this, then    iteration stops.\nmax_iterations::Int64, max number of iterations (gradient steps) taken by    the solver.\niter_diff::Vector{T}, a buffer for storing differences between subsequent   iterate values that are used for computing the step size\ngrad_diff::Vector{T}, a buffer for storing differences between gradient    values at adjacent iterates, which is used to compute the step size\niter_hist::Vector{Vector{T}}, a history of the iterates. The first entry   corresponds to the initial iterate (i.e., at iteration 0). The k+1 entry   corresponds to the iterate at iteration k.\ngrad_val_hist::Vector{T}, a vector for storing max_iterations+1 gradient   norm values. The first entry corresponds to iteration 0. The k+1 entry   correpsonds to the gradient norm at iteration k.\nstop_iteration::Int64, the iteration number that the solver stopped on.   The terminal iterate is saved at iter_hist[stop_iteration+1].\n\nConstructors\n\nBarzilaiBorweinGD(::Type{T}; x0::Vector{T}, init_stepsize::T, \n    long_stepsize::Bool, threshold::T, max_iterations::Int) where {T}\n\nConstructs the struct for the Barzilai-Borwein optimization method\n\nArguments\n\nT::DataType, specific data type used for calculations.\n\nKeyword Arguments\n\nx0::Vector{T}, initial point to start the solver at.\ninit_stepsize::T, initial step size used for the first iteration. \nlong_stepsize::Bool, flag for step size; if true, use the long version of   Barzilai-Borwein, if false, use the short version. \nthreshold::T, gradient threshold. If the norm gradient is below this,    then iteration is terminated. \nmax_iterations::Int, max number of iterations (gradient steps) taken by    the solver.\n\n\n\n\n\n","category":"type"},{"location":"api_methods/#Gradient-Descent-with-Fixed-Step-Size","page":"Methods","title":"Gradient Descent with Fixed Step Size","text":"","category":"section"},{"location":"api_methods/","page":"Methods","title":"Methods","text":"fixed_step_gd\nFixedStepGD","category":"page"},{"location":"api_methods/#OptimizationMethods.fixed_step_gd","page":"Methods","title":"OptimizationMethods.fixed_step_gd","text":"fixed_step_gd(optData::FixedStepGD{T}, progData<:AbstractNLPModel{T,S})\n    where {T,S}\n\nImplements fixed step-size gradient descent for the desired optimization problem     specified by progData.\n\nMethod\n\nThe iterates are updated according to the procedure\n\nx_k+1 = x_k - alpha f(x_k)\n\nwhere alpha is the step size, f is the objective function, and f is the      gradient function of f. \n\nArguments\n\noptData::FixedStepGD{T}, the specification for the optimization method.\nprogData<:AbstractNLPModel{T,S}, the specification for the optimization   problem. \n\nwarning: Warning\nprogData must have an initialize function that returns subtypes of AbstractPrecompute and AbstractProblemAllocate, where the latter has a grad argument. \n\n\n\n\n\n","category":"function"},{"location":"api_methods/#OptimizationMethods.FixedStepGD","page":"Methods","title":"OptimizationMethods.FixedStepGD","text":"FixedStepGD{T} <: AbstractOptimizerData{T}\n\nA structure for storing data about fixed step-size gradient descent, and the     progress of its application on an optimization problem.\n\nFields\n\nname::String, name of the solver for reference\nstep_size::T, the step-size selection for the optimization procedure\nthreshold::T, the threshold on the norm of the gradient to induce stopping\nmax_iterations::Int, the maximum allowed iterations\niter_hist::Vector{Vector{T}}, a history of the iterates. The first entry   corresponds to the initial iterate (i.e., at iteration 0). The k+1 entry   corresponds to the iterate at iteration k.\ngrad_val_hist::Vector{T}, a vector for storing max_iterations+1 gradient   norm values. The first entry corresponds to iteration 0. The k+1 entry   correpsonds to the gradient norm at iteration k\nstop_iteration, iteration number that the algorithm stopped at.   Iterate number stop_iteration is produced. \n\nConstructors\n\nFixedStepGD(::Type{T}; x0::Vector{T}, step_size::T, threshold::T, \n    max_iterations::Int) where {T}\n\nConstructs the struct for the optimizer.\n\nArguments\n\nT::DataType, specific data type for the calculations\n\nKeyword Arguments\n\nx0::Vector{T}, the initial iterate for the optimizers\nstep_size::T, the step size of the optimizer \nthreshold::T, the threshold on the norm of the gradient to induce stopping\nmax_iterations::Int, the maximum number of iterations allowed  \n\n\n\n\n\n","category":"type"},{"location":"api_methods/#Lipschitz-Approximation-(Malitsky-and-Mishchenko)","page":"Methods","title":"Lipschitz Approximation (Malitsky & Mishchenko)","text":"","category":"section"},{"location":"api_methods/","page":"Methods","title":"Methods","text":"lipschitz_approximation_gd\nLipschitzApproxGD","category":"page"},{"location":"api_methods/#OptimizationMethods.lipschitz_approximation_gd","page":"Methods","title":"OptimizationMethods.lipschitz_approximation_gd","text":"lipschitz_approximation_gd(optData::FixedStepGD{T}, progData::P where P \n    <: AbstractNLPModel{T, S}) where {T, S}\n\nImplements gradient descent with adaptive step sizes formed through a lipschitz      approximation for the desired optimization problem specified by progData.\n\nwarning: Warning\nThis method is designed for convex optimization problems.\n\nReferences(s)\n\nMalitsky, Y. and Mishchenko, K. (2020). \"Adaptive Gradient Descent without      Descent.\"      Proceedings of the 37th International Conference on Machine Learning,      in Proceedings of Machine Learning Research 119:6702-6712.     \n\nMethod\n\nThe iterates are updated according to the procedure,\n\nx_k+1 = x_k - alpha_k nabla f(x_k)\n\nwhere alpha_k is the step size and nabla f is the gradient function      of the objective function f.\n\nThe step size is computed depending on k.      When k = 0, alpha_k = optDatainit_stepsize.      When k  0, \n\nalpha_k = minleftlbrace sqrt1 + theta_k-1alpha_k-1 \n    fracVert x_k - x_k-1 VertVert nabla f(x_k) - \n    nabla f(x_k-1)Vert rightrbrace\n\nwhere theta_0 = inf and theta_k = alpha_k  alpha_k-1.\n\nArguments\n\noptData::LipschitzApproxGD{T}, the specification for the optimization method.\nprogData<:AbstractNLPModel{T,S}, the specification for the optimization   problem. \n\nwarning: Warning\nprogData must have an initialize function that returns subtypes of AbstractPrecompute and AbstractProblemAllocate, where the latter has a grad argument. \n\n\n\n\n\n","category":"function"},{"location":"api_methods/#OptimizationMethods.LipschitzApproxGD","page":"Methods","title":"OptimizationMethods.LipschitzApproxGD","text":"LipschitzApproxGD{T} <: AbstractOptimizerData{T}\n\nA structure for storing data about adaptive gradient descent     using a Lipschitz Approximation scheme (AdGD), and the progress      of its application on an optimization problem.\n\nFields\n\nname::String, name of the solver for reference\ninit_stepsize::T, the initial step size for the method\nprev_stepsize::T, step size used at iter - 1 when iter > 1.\ntheta::T, element used in the computation of the step size. See the    referenced paper for more information.\nlipschitz_approximation::T, help the lipschitz approximation used in the   computation of the step size. See the referenced paper for more information.\nthreshold::T, the threshold on the norm of the gradient to induce stopping\nmax_iterations::Int64, the maximum allowed iterations\niter_diff::Vector{T}, a buffer for storing differences between subsequent   iterate values that are used for computing the step size\ngrad_diff::Vector{T}, a buffer for storing differences between gradient    values at adjacent iterates, which is used to compute the step size\niter_hist::Vector{Vector{T}}, a history of the iterates. The first entry   corresponds to the initial iterate (i.e., at iteration 0). The k+1 entry   corresponds to the iterate at iteration k.\ngrad_val_hist::Vector{T}, a vector for storing max_iterations+1 gradient   norm values. The first entry corresponds to iteration 0. The k+1 entry   corresponds to the gradient norm at iteration k\nstop_iteration::Int64, iteration number that the algorithm stopped at.   Iterate number stop_iteration is produced. \n\nConstructors\n\nLipschitzApproxGD(::Type{T}; x0::Vector{T}, init_stepsize::T, threshold::T, \n    max_iterations::Int) where {T}\n\nConstructs the struct for the optimizer.\n\nArguments\n\nT::DataType, specific data type for the calculations\n\nKeyword Arguments\n\nx0::Vector{T}, the initial iterate for the optimizers\ninit_stepsize::T, the initial step size for the method\nthreshold::T, the threshold on the norm of the gradient to induce stopping\nmax_iterations::Int, the maximum number of iterations allowed  \n\n\n\n\n\n","category":"type"},{"location":"api_methods/#Weighted-Norm-Damping-Gradient-Method-(WNGrad)","page":"Methods","title":"Weighted Norm Damping Gradient Method (WNGrad)","text":"","category":"section"},{"location":"api_methods/","page":"Methods","title":"Methods","text":"weighted_norm_damping_gd\nWeightedNormDampingGD","category":"page"},{"location":"api_methods/#OptimizationMethods.weighted_norm_damping_gd","page":"Methods","title":"OptimizationMethods.weighted_norm_damping_gd","text":"weighted_norm_damping_gd(optData::WeightedNormDampingGD{T}, \n    progData::P where P <: AbstractNLPModel{T, S}) where {T, S}\n\nMethod that implements gradient descent with weighted norm damping step size      using the specifications in optData on the problem specified by progData.\n\nReference\n\nWu, Xiaoxia et. al. \"WNGrad: Learn the Learning Rate in Gradient Descent\". arxiv,      https://arxiv.org/abs/1803.02865\n\nMethod\n\nLet theta_k be the k^th iterate, and alpha_k be the k^th      step size. The optimization method generate iterates following\n\ntheta_k + 1 = theta_k - alpha_k nabla f(theta_k)\n\nwhere nabla f is the gradient of the objective function f.\n\nThe step size depends on the iteration number k. For k = 0, the step      size alpha_0 is the reciprocal of optData.init_norm_damping_factor.      For k  0, the step size is iteratively updated as\n\nalpha_k = left\nfrac1alpha_k-1 + Vertdot F(theta_k)Vert_2^2 alpha_k-1\nright^-1\n\nwarning: Warning\nWhen alpha_0  Vertdot F(theta_0)Vert_2^-1 and a globally Lipschitz smooth objective function is used, then the method is guaranteed to find an epsilon-stationary point. It is recommended then that  optData.init_norm_damping_factor exceed Vertdot F(theta_0)Vert_2.\n\nArguments\n\noptData::WeightedNormDampingGD{T}, specification for the optimization algorithm.\nprogData::P where P <: AbstractNLPModel{T, S}, specification for the problem.\n\nwarning: Warning\nprogData must have an initialize function that returns subtypes of AbstractPrecompute and AbstractProblemAllocate, where the latter has a grad argument. \n\n\n\n\n\n","category":"function"},{"location":"api_methods/#OptimizationMethods.WeightedNormDampingGD","page":"Methods","title":"OptimizationMethods.WeightedNormDampingGD","text":"WeightedNormDampingGD{T} <: AbstractOptimizerData{T}\n\nA mutable struct that represents gradient descent using the weighted-norm      damping step size. It stores the specification for the method and records      values during iteration.\n\nFields\n\nname::String, name of the optimizer for recording purposes\ninit_norm_damping_factor::T, initial damping factor. This value's reciprocal    will be the initial step size.\nthreshold::T, norm gradient tolerance condition. Induces stopping when norm    is at most threshold.\nmax_iterations::Int64, max number of iterates that are produced, not    including the initial iterate.\niter_hist::Vector{Vector{T}}, store the iterate sequence as the algorithm    progresses. The initial iterate is stored in the first position.\ngrad_val_hist::Vector{T}, stores the norm gradient values at each iterate.    The norm of the gradient evaluated at the initial iterate is stored in the    first position.\nstop_iteration::Int64, the iteration number the algorithm stopped on. The    iterate that induced stopping is saved at iter_hist[stop_iteration + 1].\n\nConstructors\n\nWeightedNormDampingGD(::Type{T}; x0::Vector{T}, init_norm_damping_factor::T, \n    threshold::T, max_iterations::Int64) where {T}\n\nConstructs an instance of type WeightedNormDampingGD{T}.\n\nArguments\n\nT::DataType, type for data and computation\n\nKeyword Arguments\n\nx0::Vector{T}, initial point to start the optimization routine. Saved in   iter_hist[1].\ninit_norm_damping_factor::T, initial damping factor, which will correspond   to the reciprocoal of the initial step size. \nthreshold::T, norm gradient tolerance condition. Induces stopping when norm    at most threshold.\nmax_iterations::Int64, max number of iterates that are produced, not    including the initial iterate.\n\n\n\n\n\n","category":"type"},{"location":"api_methods/#Index","page":"Methods","title":"Index","text":"","category":"section"},{"location":"api_methods/","page":"Methods","title":"Methods","text":"","category":"page"},{"location":"problems/quasilikelihood_estimation/#Quasi-likelihood-Estimation","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"","category":"section"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"Quasi-likelihood estimation was first introduced by  Wedderburn (1974) as a way of estimating regression coefficients when the underlying probability model generating the data is hard to identify,  or when the data is not explained well by common approaches, such as generalized linear models.  We now briefly provide background information on the method, followed by some examples that are implemented in our package.","category":"page"},{"location":"problems/quasilikelihood_estimation/#Quasi-likelihood-Setting-and-Estimation-Methodology","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Setting and Estimation Methodology","text":"","category":"section"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"Quasi-likelihood estimation, following Wedderburn's description, assumes that the response variables, y_i in mathbbR, for i = 1n, being either discrete or continuous, are independently collected from a distribution that is only partially known. Specifically, for covariate vectors, x_i in mathbbR^p, i = 1n, and a vector theta^star in mathbbR^p the model assumes the following relationship between y_i x_i and theta^star.","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"(Mean Relationship) The expected value of each observation, mathbbEy_i  x_i theta^star = mu_i, satisfies mu_i = mu(x_i^intercal theta^star) for a known function mu  mathbbR to mathbbR. Typically, g is selected to be invertible.\n(Variance Relationship) The variance of each data point satisfies mathbbVy_i  x_i theta^star = V(mu(x_i^intercal theta^star)) for a known non-negative function V  mathbbR to mathbbR.","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"Estimation of theta^star proceeds by combining these two components to form the quasi-likelihood objective,","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"min_theta F(theta) = min_theta -sum_i=1^n int_c_i^mu(x_i^intercal theta) fracy_i - mV(m)dm","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"note: Note\nSince the mean and variance function can be arbitrarily selected, F(theta) might not correspond to any known likelihood function, hence the name quasi-likelihood. Furthermore, the objective function F(theta) might not have a closed form expression, requiring numerical integration techniques to evaluate.  Owing to the structure of the objective however, the fundamental theorem of calculus can be used to compute the gradient. This potentially makes the objective more expensive to compute than the gradient, an atypical scenario for optimization.","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"Having presented some background on quasi-likelihood estimation, we now present a use case of the framework for a special case in semi-parametric regression.","category":"page"},{"location":"problems/quasilikelihood_estimation/#Example:-Semi-parametric-Regression-with-Heteroscedasticity-Errors","page":"Quasi-likelihood Estimation","title":"Example: Semi-parametric Regression with Heteroscedasticity Errors","text":"","category":"section"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"Suppose that observations, y_i in mathbbR, i = 1  n, are independent  and are associated with covariate vectors x_i in mathbbR^p, i = 1n.  Furthermore, suppose (x_i y_i) satisfy the following relationship","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"y_i = mu(x_i^intercal theta^star) + V( mu(x_i^intercal theta^star) )^12 epsilon_i","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"for a function mumathbbR to mathbbR, a non-negative function  V  mathbbR to mathbbR, and a vector theta^star in mathbbR^p. Here, epsilon_i are independent realization from a distribution with a mean and variance of 0 and 1, respectively, but whose exact form cannot be fully specified. ","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"This model is a special form of semi-parametric regression with heteroscedastic errors, also  satisfying the requirements of the quasi-likelihood estimation framework. Indeed, the mean and variance relationships for quasi-likelihood are satisfied by checking the expected value and variance of y_i using the statistical model above.","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"Below, we provide a list of variance functions that lead to the quasi-likelihood objective being hard to analytically integrate (if not impossible), some of which appear in literature.","category":"page"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"Let V  mathbbR to mathbbR be defined as V(mu) = 1 + mu + sin(2pimu). See Section 4 of Lanteri et al. (2023).\nLet V  mathbbR to mathbbR be defined as V(mu) = (mu^2)^p + c for c in mathbbR_ 0 and p in mathbbR_5. See for example variance stabilization transformations.\nLet V  mathbbR to mathbbR be defined as V(mu) = exp(-((mu - c)^2)^p) for c in mathbbR and p in mathbbR_5.\nLet V  mathbbR to mathbbR be defined as V(mu) = log( ((mu - c)^2)^p + 1) for c in mathbbR and p in mathbbR_5.","category":"page"},{"location":"problems/quasilikelihood_estimation/#Model-Implementation","page":"Quasi-likelihood Estimation","title":"Model Implementation","text":"","category":"section"},{"location":"problems/quasilikelihood_estimation/#References","page":"Quasi-likelihood Estimation","title":"References","text":"","category":"section"},{"location":"problems/quasilikelihood_estimation/","page":"Quasi-likelihood Estimation","title":"Quasi-likelihood Estimation","text":"Lanteri, A.; Leorato, S.; Lopez-Fidalgo, J. and Tommasi, C. (2023). Designing to detect heteroscedasticity in a regression model. Journal of the Royal Statistical Society 85, 315–326.\n\n\n\nWedderburn, R. W. (1974). Quasi-Likelihood Functions, Generalized Linear Models, and the Gauss—Newton Method. Biometrika 61, 439–447.\n\n\n\n","category":"page"},{"location":"#Overview","page":"Overview","title":"Overview","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"OptimizationMethods is a  Julia library for implementing and comparing optimization methods with a focus on problems arising in data science. The library is primarily designed to serve those researching optimization  methods for data science applications. Accordingly, the library is not implementing highly efficient versions of these methods, even though we do our best to make preliminary efficiency optimizations to the code.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"There are two primary components to this library.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"Problems, which are implementations of optimization problems primarily   arising in data science. At the moment, problems follow the guidelines   provided by    NLPModels. \nMethods, which are implementations of important optimization methods   that appear in the literature. ","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"The library is still in its infancy and will continue to evolve rapidly. To understand how to use the library, we recommend looking in the examples directory to see how different problems are instantiated and how optimization methods can be applied to them. We also recommend looking at the docstring for specific problems and methods for additional details.","category":"page"},{"location":"#Manual","page":"Overview","title":"Manual","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"The manual section includes descriptions of problems and methods that require a bit more explanation than what is appropriate for in a docstring.","category":"page"},{"location":"#API","page":"Overview","title":"API","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"The API section contains explanations for all problems and methods available in  the library. This is a super set of what is contained in the manual. ","category":"page"}]
}
