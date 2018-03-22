
export normcols, normrows, rank, qrfact1
export tolabsdefault, tolreldefault, adjustsize
export QRFactorization, QRWrapper

const QRFactorization = Union{QRPivoted,
                              LinearAlgebra.QRCompactWY,
                              SuiteSparse.SPQR.QRSparse}

struct QRWrapper{F<:Union{QRPivoted,LinearAlgebra.QRCompactWY,SuiteSparse.SPQR.QRSparse}}
    parent::F
end

"""
    calculate norms all columns of a sparse matrix
"""
function normcols(A::SparseMatrixCSC{T}) where T<:Number
    i = 1
    TT = typeof(sqrt(one(eltype(A))))
    z = zero(T)
    sum = z
    r = Vector{TT}(undef, A.n)
    for j = 1:A.n
        k = A.colptr[j+1]
        while i < k
            sum += abs2(A.nzval[i])
            i += 1
        end
        r[j] = sum
        sum = z
        end
        sqrt.(real(r))
end

function normcols(A::AbstractMatrix{T}) where T<:Number
    maximum(norm(view(A, :, i)) for i in 1:size(A, 2))
end

"""
    calculate norms of all rows of a sparse matrix
"""
function normrows(A::SparseMatrixCSC{T}) where T<:Number
    TT = typeof(sqrt(one(eltype(A))))
    r = zeros(TT, size(A, 1))
    for i = 1:nnz(A)
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
function tolabsdefault(A::AbstractMatrix{T}) where T<:Number
    20maximum(normcols(A)) * eps(real(T)) * max(size(A)...)
end

"""
    default tolerance relative for rank-calculations
"""
function tolreldefault(A)
    T = eltype(A)
    sqrt(size(A,1)) * eps(real(T)) * 20
end

"""
    absolute tolerance as a combination of tolrel and tolabs
"""
function tolerance(A::AbstractMatrix, pivot::Bool,
                   tolrel::AbstractFloat, tolabs::AbstractFloat)

    tolabs >= 0 || throw(ArgumentError("absolute tolerance must not be negative"))
    tolrel >= 0 || throw(ArgumentError("relative tolerance must not be negative"))
    if tolrel == 0 == tolabs && pivot
        tolrel = tolreldefault(A)
    end
    tolrel == 0 ? tolabs : max(tolrel * maximum(normcols(A)), tolabs)
end

"""
    rank of a QR-factorization of a matrix
"""
function LinearAlgebra.rank(F::Union{QRFactorization,QRWrapper};
                            tolrel::AbstractFloat=0.0, tolabs::AbstractFloat=0.0)

    FR = F.R
    tol = tolerance(FR, true, tolrel, tolabs)
    count(abs2.(diag(FR)) .> tol^2)
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
    elseif A isa SparseMatrixCSC &&
        ( n0 <= n && ( m > m0 || maximum(view(A.rowval, 1:nnz(A))) <= m < m0 ))
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

Base.show(io::IO, qr::QRWrapper) = show(io, qr.parent)
Base.propertynames(qr::QRWrapper, private::Bool) = (:R, :Q, :pcol, :prow)
import Base.getproperty

function getproperty(qr::QRWrapper{<:SuiteSparse.SPQR.QRSparse}, d::Symbol)
    if d == :R || d == :Q || d == :pcol || d == :prow
        getproperty(qr.parent, d)
    elseif d == :p
        qr.parent.pcol
    else
        getfield(qr, d)
    end
end
function getproperty(qr::QRWrapper{<:QRPivoted}, d::Symbol)
    if d == :R || d == :Q || d == :p || d == :P
        getproperty(qr.parent, d)
    elseif d == :pcol
        getproperty(qr.parent, :p)
    elseif d == :prow
        collect(1:size(qr.parent.Q, 1))
    else
        getfield(qr, d)
    end
end
function getproperty(qr::QRWrapper{<:LinearAlgebra.QRCompactWY}, d::Symbol)
    if d == :R || d == :Q
        getproperty(qr.parent, d)
    elseif d == :pcol
        collect(1:size(qr.parent.R, 2))
    elseif d == :prow
        collect(1:size(qr.parent.Q, 1))
    else
        getfield(qr, d)
    end
end

Base.size(qrw::QRWrapper, arg...) = size(qrw.parent, arg...)

function qrfact1(A::SparseMatrixCSC{<:Union{ComplexF64,Float64}};
                 tolrel::AbstractFloat=0.0, tolabs::AbstractFloat=0.0, pivot=true)

    tol = tolerance(A, pivot, tolrel, tolabs)
    QRWrapper(qrfact(A, tol=tol))
end

function qrfact1(A::SparseMatrixCSC;
                 tolrel::AbstractFloat=0.0, tolabs::AbstractFloat=0.0, pivot=true)
        
    throw(ArgumentError("qrfact does not support $(typeof(A))"))
end

function qrfact1(A::AbstractMatrix;
                 tolrel::AbstractFloat=0.0, tolabs::AbstractFloat=0.0, pivot=true)

    tol = pivot ? tolerance(A, pivot, tolrel, tolabs) : 0.0
    if tol > 0 && !pivot
        throw(ArgumentError("tol > 0 requires pivot=true"))
    end
    qr = qrfact(A, Val(pivot))
    if tol > 0
        m = size(qr.factors, 1)
        k = rank(qr, tolrel=tolrel, tolabs=tolabs)
        for i = k+1:length(qr.τ)
            qr.τ[i] = 0
            qr.factors[i:m,i] = 0
        end
    end
    QRWrapper(qr)
end

function qrfact1(A::Union{Adjoint, Transpose};
                 tolrel::AbstractFloat=0.0, tolabs::AbstractFloat=0.0, pivot=true)

    qrfact1(copy(A), tolrel=tolrel, tolabs=tolabs)
end

