
export pinvfact, PInv, PInvFact, rank, adjoint

abstract type PInv{T} <: Factorization{T} end

mutable struct PInvFact{T} <: PInv{T}
    m::Int
    n::Int
    k::Int
    F::Factorization{T}
    G::Factorization{T}
end

Basepropertynames(PI::PInvFact) = (:R, :Q, :prow, :pcol)

import LinearAlgebra: rank, adjoint
rank(pi::PInv) = rank(pi.F)
adjoint(pi::PInv) = Adjoint(pi)

function pinvfact(A::SparseMatrixCSC; tol=SuiteSparse.SPQR._default_tol(A))
    m, n = size(A)
    if m <= 100000000000000000000 #n
        F = qrfact(A, tol=tol)
        k = rank(F)
        G = qrfact(hardadjoint(adjustsize(F.R, k, n)), tol=tol/2)
        @assert k == rank(G) "rank defect during second QR factorization"
        PInvFact(m, n, k, F, G)
    else
        adjoint(pinvfact(hardadjoint(A), tol=tol))
    end
end

Base.size(PI::PInvFact) = (PI.m, PI.n)
Base.size(PIA::Adjoint{<:Number,<:PInvFact}) = size(PIA.parent)

import Base.*

function (*)(PI::PInv{T}, A::StridedVecOrMat{T}) where T
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

function *(PIT::Adjoint{S}, A::StridedVecOrMat{T}) where {T,S<:PInv{T}}
    PI = PIT.parent
    m, n = size(PI)
    if n != size(A, 1)
        throw(ArgumentError("Dimension mismatch"))
    end
    k = PI.k
    F = PI.F
    G = PI.G
    nr = size(A, 2)
    tmpn = similar(A)
    tmpn[invperm(F.pcol),:] = A
    x1 = view(tmpn, 1:k, :)
    lmul!(G.Q', x1)
    x2 = F.Q * ( G.R \ x1 )
    x2[F.prow,:]
end
        
