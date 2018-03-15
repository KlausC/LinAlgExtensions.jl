
export pinvfact, PInv, PInvFact, rank, adjoint

abstract type PInv{T} <: Factorization{T} end

mutable struct PInvFact{T} <: PInv{T}
    m::Int
    n::Int
    k::Int
    F::Factorization{T}
    R12::AbstractMatrix{T}
    C::Factorization{T}
end

Basepropertynames(PI::PInvFact) = (:R, :Q, :prow, :pcol)

if VERSION < v"0.7.0-DEV"
    struct Adjoint{T,S<:Factorization{T}}
        parent::S
    end
end

import LinearAlgebra: rank, adjoint
rank(pi::PInv) = rank(pi.F)
adjoint(pi::PInv) = Adjoint(pi)

function pinvfact(A::SparseMatrixCSC; tol=SuiteSparse.SPQR._default_tol(A))
    m, n = size(A)
    if m <= 1000 # n
        F = qrfact(A, tol=tol)
        k = rank(F)
        R1 = view(F.R, 1:k, 1:k)
        R2 = view(F.R, 1:k, k+1:n)
        R12 = R1 \ R2
        if n <= 2k
            # R2 is tall
            C = cholfact(R12'R12; shift=1.0)
        else
            C = cholfact(R12*R12'; shift=1.0)
        end
        PInvFact(m, n, k, F, R12, C)
    else
        Adjoint(pinvfact(adjoint.(permutedims(A)), tol=tol))
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
    R1 = view(F.R, 1:k, 1:k)
    R12 = PI.R12
    C = PI.C
    nr = size(A, 2)
    tmpm = similar(A, ntuple(i-> i == 1 ? m : nr, ndims(A)))
    tmpm[:,:] = view(A, F.prow, :)
    lmul!(F.Q', tmpm)
    y1 = R1 \ view(tmpm, ntuple(i-> i == 1 ? (1:k) : (1:nr), ndims(A))...)
    y2 = R12' * y1
    if n <= 2k
        y2 = C \ y2
    else
        y2 -= R12' * ( C \ (R12 * y2) )
    end
    y1 -= R12 * y2
    [y1; y2][invperm(F.pcol), :]
end

function *(PIT::Adjoint{S}, A::StridedVecOrMat{T}) where {T,S<:PInv{T}}
    PI = PIT.parent
    m, n = size(PI)
    if n != size(A, 1)
        throw(ArgumentError("Dimension mismatch"))
    end
    k = PI.k
    F = PI.F
    R1 = view(F.R, 1:k, 1:k)
    R12 = PI.R12
    C = PI.C
    nr = size(A, 2)
    tmpn = similar(A)
    tmpn[invperm(F.pcol),:] = A
    x1 = view(tmpn, 1:k, :)
    x2 = view(tmpn, k+1:n, :) - R12' * x1
    if n <= k
        x1 += R12 * ( C \ x2 )
    else
        x1 -= R12 * ( x2 - (R12' * ( C \ ( R12 * x2) ) ) )
    end
    tmpm = similar(A, ntuple(i-> i == 1 ? m : nr, ndims(A)))
    tmpm[F.prow,:] = F.Q * ( R1' \ x1 )
    tmpm
end
        
