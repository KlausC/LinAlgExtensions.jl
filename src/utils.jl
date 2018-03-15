
export normcols, normrows, rank, tolabs, tolrel

if VERSION >= v"0.7.0-DEV"
    get_R(F::Factorization) = F.R
    get_Q(F::Factorization) = F.Q
else
    get_R(F::Factorization) = F[:R]
    get_Q(F::Factorization) = F[:Q]
end

"""
    calculate norms all columns of a sparse matrix
"""
function normcols(A::SparseMatrixCSC{T}) where T<:Number
    i = 1
    TT = typeof(sqrt(one(eltype(A))))
    z = zero(T)
    sum = z
    r = Vector{TT}(length(A.colptr)-1)
    for j = 1:length(r)
        k = A.colptr[j+1]
        while i < k
            sum += abs2(A.nzval[i])
            i += 1
        end
        r[j] = sum
        sum = z
        end
    sqrt.(r)
end

function normcols(A::AbstractMatrix{T}) where T<:Number
    maximum(norm(view(A, :, i)) for i in 1:size(A, 2))
end

"""
    calculate norms rows of a sparse matrix
"""
function normrows(A::SparseMatrixCSC{T}) where T<:Number
    TT = typeof(sqrt(one(eltype(A))))
    r = zeros(TT, size(A, 1))
    for i = 1:length(A.rowval)
        j = A.rowval[i]
        r[j] += abs2(A.nzval[i])
    end
    sqrt.(r)
end

function normrows(A::AbstractMatrix{T}) where T<:Number
    maximum(norm(view(A, i, :)) for i in 1:size(A, 1))
end

"""
    default tolerance absolute for qr-calculations
"""
function tolabs(A::AbstractMatrix{T}) where T<:Number
    20maximum(normcols(A)) * eps(real(T)) * max(size(A)...)
end

"""
    default tolerance relative for rank-calculations
"""
function tolrel(A)
    T = eltype(A)
    sqrt(size(A,1)) * eps(real(T))
end

"""
    rank of a QR-factorization of a matrix
"""
function LinearAlgebra.rank(F::Factorization, tol=tolrel(F))
    FR = get_R(F)
    tt = (tol * maximum(normcols(FR)))^2
    count(abs2.(diag(get_R(F))) .> tt)
end

