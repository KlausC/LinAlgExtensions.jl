
export pinvfact, PInv, PInvFact, rank, adjoint

abstract type PInv{T} <: Factorization{T} end

"""
Structure to hold pseudo-inverse factorization.
"""
mutable struct PInvFact{T} <: PInv{T}
    m::Int                  # number of rows of input matrix
    n::Int                  # number of columns
    k::Int                  # rank
    F::QRWrapper   # first QR
    G::QRWrapper   # second QR (in case k < n)
end

Base.propertynames(PI::PInvFact) = (:R, :Q, :prow, :pcol)

import LinearAlgebra: rank, adjoint
rank(psi::PInv) = rank(psi.F)
rank(psi::Adjoint{<:Number,<:PInv}) = rank(psi.parent)

adjoint(psi::PInv) = Adjoint(psi)
function Base.show(io::IO, mime::MIME, apsi::Adjoint{T,<:PInv}) where T 
    show(io, mime, typeof(apsi))
    show(io, mime, apsi.parent)
end

"""
    pinvfact(A::AbstractMatrix; tol)

Create a Moore-Penrose pseudo-inverse factorization of input matrix A.
A Moore-Penrose pseudo-inverse is represented as one or two pivoted
QR-factorizations. The first factorization determines also the rank of the
matrix. The second one is applied to the transposed of the results of the
first factorization.

The usual way to calculate a pseudo inverse (pinv) makes use of a "SVD"
(singular value decomposition) of `A`, which in turn requires an eigenvalue
decomposition of `A'A`. As the pseudo-inverse itself is typically dense, also
if the input matrix is sparse, it is expensive to calculate it as a matrix.
This implementation stores only the results of two QR-decompositions, which
remain sparse, which makes the method suitable for big sparse matrices as well.

Usage:
psi = pinvfact(A)
x = psi * b
"""
function pinvfact(A::AbstractMatrix; tolrel::AbstractFloat=0.0, tolabs::AbstractFloat=0.0)
    m, n = size(A)
    # TODO decide if that makes sense:
    # 1. probable use case m >= n
    # 2. better error bounds on rank-determining pivots if m < n
    if m >= n
        F = qrfact1(A, tolrel=tolrel, tolabs=tolabs)
        k = rank(F, tolrel=tolrel, tolabs=tolabs)
        G = qrfact1(copy(adjoint(adjustsize(F.R, k, n))), tolrel=tolrel, tolabs=tolabs)
        @assert k == rank(G) "rank defect during second QR factorization"
        PInvFact{eltype(A)}(m, n, k, F, G)
    else
        adjoint(pinvfact(copy(adjoint(A)), tolrel=tolrel, tolabs=tolabs))
    end
end

Base.size(PI::PInvFact, args...) = size(PI.F, args...)
Base.size(PIA::Adjoint{<:Number,<:PInv}) = reverse(size(PIA.parent))

import Base.\

"""
    Apply the pseudo-inverse factorization to the rhs `A`.
"""
function \(PI::PInv{T}, A::StridedVecOrMat{T}) where T
    m, n = size(PI)
    if m != size(A, 1)
        throw(ArgumentError("Dimension mismatch"))
    end
    k = PI.k
    F = PI.F
    G = PI.G
    nr = size(A, 2)
    tmpm = similar(A, ntuple(i-> i == 1 ? m : nr, ndims(A)))
    tmpm[:,:] = view(A, F.prow, :)
    lmul!(F.Q', tmpm)
    tmpx = view(tmpm, ntuple(i-> 1:(i == 1 ? k : nr), ndims(A))...)
    tmpx = view(tmpx, G.pcol, :)
    y1 = adjustsize((G.R' \ tmpx), n, nr)
    y1 = (G.Q * y1)[invperm(G.prow), :]
    y1[invperm(F.pcol), :]
end

"""
    Apply the adjoint of pseudoinverse factorization to rhs.
"""
function \(PIT::Adjoint{<:Number,S}, A::StridedVecOrMat{T}) where {T,S<:PInv{T}}
    PI = PIT.parent
    m, n = size(PI)
    if n != size(A, 1)
        throw(ArgumentError("Dimension mismatch"))
    end
    k = PI.k
    F = PI.F
    G = PI.G
    nr = size(A, 2)
    x1 = similar(A)
    x1[invperm(F.pcol),:] = A[G.prow, :]
    lmul!(G.Q', x1)
    x1[invperm(G.pcol),:] = G.R \ x1[1:k,:]
    x2 = F.Q * adjustsize(x1, k, nr)
    x2[F.prow,:]
end
        
