
export normcols, normrows, rank, tolabs, tolrel, hardadjoint, adjustsize

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

function hardadjoint(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    acolptr = zeros(Ti, A.m+1)
    arowval = similar(A.rowval)
    anzval = similar(A.nzval)
    acolptr[1] = 1
    nnz = length(anzval)
    for j = 1:nnz
        acolptr[A.rowval[j]+1] += 1
    end
    for k = 1:A.m
        acolptr[k+1] += acolptr[k]
    end
    for k = 1:A.n
        for j = A.colptr[k]:A.colptr[k+1]-1
            r = A.rowval[j]
            i = acolptr[r]
            arowval[i] = k
            anzval[i] = adjoint(A.nzval[j])
            acolptr[r] += 1
        end
    end
    fill!(acolptr, zero(Ti))
    acolptr[1] = one(Ti)
    for j = 1:nnz
        acolptr[A.rowval[j]+1] += 1
    end
    for k = 1:A.m
        acolptr[k+1] += acolptr[k]
    end
    SparseMatrixCSC(A.n, A.m, acolptr, arowval, anzval)
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

"""
    `adjustsize(A::Matrix, m:Integer, n::Integer)`

Return a Matrix with `m` rows and `n` columns. If `m > size(A, 1)` new zero rows are added,
if `n > size(A, 2)` zero columns are added. The left upper corner of the resulting
matrix is filled with the corresponding elements of `A`.
"""
function adjustsize(A::AbstractMatrix, m::Integer=size(A,1), n::Integer=size(A,2))
    
    m0, n0 = size(A)
    if n == n0 && m == m0
        A
    elseif A isa SparseMatrixCSC && ( n0 <= n && ( m > m0 || maximum(A.rowval) <= m < m0 ))
        if n == n0
            acolptr = A.colptr
        else
            acolptr = resize!(copy(A.colptr), n+1)
            fill!(view(acolptr, n0+2:n+1), acolptr[n0+1])
        end  
        SparseMatrixCSC(m, n, acolptr, A.rowval, A.nzval)
    elseif m <= m0 && n <= n0
        A[1:m,1:n]
    else
        B = similar(A, m, n)
        fill!(B, 0)
        m0 = min(m, m0)
        n0 = min(n, n0)
        B[1:m0,1:n0] = A[1:m0,1:n0]
        B
    end
end

