
export randorth, randrealsv, randsparse

"""
    Construct random orthogonal matrix.
"""
function randorth(T::Type{<:Real}, m::Integer, n::Integer=m)
    tr = m < n
    n, m = minmax(m, n)
    A = randn(T, m, n)
    F = qrfact(A, Val(true))
    Q = copy(F.Q)[:,1:n]
    tr ? copy(Q') : Q
end
randorth(m::Integer, n::Integer=m) = randorth(Float64, m, n)

"""
    Construct random Matrix with given singular values
"""
function randrealsv(sv::AbstractVector{T}, m::Integer=0, n::Integer=0) where T<:Real
    TT = typeof(sqrt(one(eltype(sv))))
    ls = length(sv)
    m = max(m, ls)
    n = max(n, ls)
    Matrix((( randorth(TT, n, ls) * Diagonal(sv)) * randorth(TT, m, ls)')')
end

"""
    randsparse(m, n, r, p)

Construct sparse matrix with given dimensions and rank and p
"""
function randsparse(m::Integer, n::Integer, k::Integer, p::AbstractFloat)
    k <= min(m, n) || throw(ArgumentError("rank $k cannot be > min(m,n) = $(min(m, n))"))
    0 <= p <= 1 || throw(ArgumentError("probability $p in not in unit interval"))
    
    if m < n
        return copy(adjoint(randsparse(n, m, k, p)))
    end
    
    A = sprandn(m, k, p)

    if n > k
        e1(x) = x == zero(x) ? one(x) : expm1(x) / x # `(exp(x) - 1) / x` exactly
        α = log(one(p) - p)
        pp = e1(α / k) / e1(α) / k
        B = sprandn(k, n-k, pp)
        A = [A A * B]
        A[:,randperm(n)]
    else
        A
    end
end

