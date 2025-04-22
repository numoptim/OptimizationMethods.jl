# Date: 2025/04/11
# Author: Christian Varner
# Purpose: Implement a function to solve a linear system
# of the form Lx = b, where L is an upper triangle matrix or
# lower triangular matrix

"""
    upper_triangle_solve!(b::Vector{T}, U::AbstractMatrix) where {T}

Solve the linear system ``Ux = b``, where `U` is an upper triangular
matrix. The function overwrites `b` with the solution of the linear system.

# Arguments

- `b::Vector{T}`, constant vector in linear system. This vector will be 
    overwritten with the solution.
- `U::AbstractMatrix`, coefficient matrix in the linear system. Should be
    upper triangular

"""
function upper_triangle_solve!(b::Vector{T}, U::AbstractMatrix) where {T}
    n = size(U)[1]
    for i in n:-1:1
        for j in i+1:n
            b[i] -= U[i, j] * b[j]
        end
        b[i] /= U[i,i]
    end
    
    return nothing
end

"""
    lower_triangle_solve!(b::Vector{T}, L::AbstractMatrix) where {T}

Solve the linear system ``Lx = b``, where `L` is a lower triangular matrix.
The solution to the linear system will be stored in the vector `b`. 

!!! note  
    The vector `b` is overwritten by this function.

# Arguments

- `b::Vector{T}`, constant vector in the system of equations. The contents
    of this vector will be overwritten.
- `L::AbstractMatrix`, coefficient matrix of the linear system. Should be a
    lower triangular matrix.
"""
function lower_triangle_solve!(b::Vector{T}, L::AbstractMatrix) where {T}
    n = size(L)[1]
    for i in 1:n
        for j in 1:(i-1)
            b[i] -= L[i, j] * b[j]
        end
        b[i] /= L[i,i]
    end

    return nothing
end

"""
"""
function cholesky_and_solve(b::Vector{T}, A::AbstractMatrix) where {T}

    # compute cholesky and check success
    C = cholesky(Hermitian(A); check = false)
    if !issuccess(C)
        return false
    end

    # solve -- solution stored in b if cholesky was successful
    lower_triangle_solve!(b, C.U')
    upper_triangle_solve!(b, C.U)

    return true
end