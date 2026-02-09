module TestQuasiLikelihood

using Test, ForwardDiff, OptimizationMethods, Random, LinearAlgebra

function v(μ, x)
    p = x[1]
    c = x[2]
    return c * μ * (1 - μ)^(p)
end

function ∂v_∂μ(μ::T, x::Vector{T}) where {T}
    p = x[1]
    c = x[2]
    return c * ((1 - μ)^(p) - p * μ * (1 - μ)^(p - 1))
end

function ∂v_∂x(μ, x)
    p = x[1]
    c = x[2]
    return [c * log(1-μ) * (1-μ)^(p) * μ, μ*(1-μ)^p]
end

function ∂²v_∂x²(μ::T, x::Vector{T}) where {T}
    p = x[1]
    c = x[2]
    A = zeros(T, 2, 2)
    A[1, 1] = c * log(1-μ)^2 * (1-μ)^p * μ
    A[1, 2] = log(1-μ) * (1-μ)^(p) * μ
    A[2, 1] = log(1-μ) * (1-μ)^(p) * μ
    A[2, 2] = 0.0 
    return A
end


@testset "SOSVarianceObjective --- Testing Initialization" begin

    # test initialization
    dim = 50    
    let dim = dim
        design = randn(dim, dim)
        θ = randn(dim)
        resp = design * θ

        mean = OptimizationMethods.logistic.(design * θ) 
        ψ = [1., 1.]

        progData = OptimizationMethods.SOSVarianceObjective(
                    mean,
                    resp,
                    v,
                    ∂v_∂x,
                    ∂²v_∂x²,
                    ψ
                )
        @test sum(progData.response .== resp) == dim
        @test sum(progData.μ .== mean) == dim
    end
end

@testset "SOSVarianceObjective --- Testing Gradient and Hessian" begin

    # test gradient
    let dim = 50
        design = randn(dim, dim)
        θ = randn(dim)
        resp = rand(dim)
        mean = OptimizationMethods.logistic.(design * θ) 
        ψ = [1., 1.]
        progData = OptimizationMethods.SOSVarianceObjective(
                mean,
                resp,
                v,
                ∂v_∂x,
                ∂²v_∂x²,
                ψ
            )
        precomp, store = OptimizationMethods.initialize(progData)

        function obj(x::Vector{T}) where {T}
            o = 0
            for i in 1:dim
                o += .5*( (resp[i] - mean[i])^2 - v(mean[i], x) )^2
            end
            return o
        end

        function ∇obj(x::Vector{T}) where {T}
            g = zeros(T, 2)
            for i in 1:dim
                g .+= ( (resp[i] - mean[i])^2 - v(mean[i], x) ) .* (-∂v_∂x(mean[i], x))
            end
            return g
        end

        # test objective
        x = rand(2)
        @test obj(x) ≈ OptimizationMethods.obj(progData, x)

        # test gradient
        x = rand(2)
        OptimizationMethods.grad!(progData, precomp, store, x)
        @test store.grad ≈ ForwardDiff.gradient(obj, x)
        @test store.grad ≈ ∇obj(x)
        
        # test hessian
        OptimizationMethods.hess!(progData, precomp, store, x)
        h = ForwardDiff.jacobian(∇obj, x)
        @test store.hess ≈ h
    end

end

end # end module