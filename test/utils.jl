
import LinAlgExtensions: tolerance, tolreldefault, tolabsdefault
import Base: getproperty


srand(1)
let X = [spzeros(5,1) sprandn(5, 6, 0.5)], m = size(X, 1), n = size(X, 2)

stype(X) = X isa AbstractSparseMatrix ? "sparse" : "dense"

@testset "$(stype(A))" for A in (X, Matrix(X)) 
    @test normrows(A) ≈ sqrt.(diag(A*A'))
    @test normcols(A) ≈ sqrt.(diag(A'A))
end

ncx = maximum(normcols(X))

@test tolreldefault(1.0+0im) ≈ eps() * 20
@test tolabsdefault(X) ≈ ncx * eps() * n * 20

@test tolerance(X, true, 0.0, 0.0) == tolreldefault(X) * ncx
@test tolerance(X, true, 1e-5, 0.0)  == 1e-5 * ncx
@test tolerance(X, true, 0.0, 1e-5)  == 1e-5
@test tolerance(X, true, 1e-5, 1e-4)  == 1e-4
@test tolerance(X, true, 1e-5, 1e-20)  == 1e-5 * ncx
@test tolerance(X, false, 0.0, 0.0) == 0
@test tolerance(X, false, 1e-5, 0.0)  == 1e-5 * ncx
@test tolerance(X, false, 0.0, 1e-5)  == 1e-5
@test tolerance(X, false, 1e-5, 1e-4)  == 1e-4
@test tolerance(X, false, 1e-5, 1e-20)  == 1e-5 * ncx


@testset "$(stype(A))" for A in (X, Matrix(X))
    @test adjustsize(A, 5, 7) === A
    @test adjustsize(A, 7, 7) == [A; zeros(2, 7)] 
    @test adjustsize(A, 5, 9) == [A zeros(5, 2)] 
    @test adjustsize(A, 7, 10) == [A zeros(5, 3); zeros(2, 10)] 
    @test adjustsize(A, 4, 10) == [A[1:4,:] zeros(4, 3)] 
    @test adjustsize(A, 8, 2) == [A[:,1:2]; zeros(3, 2)] 
    @test adjustsize(A, 4, 5) == A[1:4,1:5]
end

@testset "$(stype(A)) pivot=$piv" for A in (X, Matrix(X)), piv in (true, false)
    qrf = qrfactors(A, pivot=piv)
    @test size(qrf) == size(A)
    @test size(qrf.Q) == (size(A, 1), size(A, 1))
    @test size(qrf.R) == (min(size(A)...), size(A, 2))
    @test isperm(qrf.pcol)
    @test length(qrf.pcol) == n
    @test isperm(qrf.prow)
    @test length(qrf.prow) == m
end

@test_throws ArgumentError qrfactors(Float16.(X))
qrf1 = qrfactors(X')
qrf2 = qrfactors(copy(X'))
@testset "adjoint $d" for d in (:Q, :R, :pcol, :prow)
    @test getproperty(qrf1, d) == getproperty(qrf2, d)
end
end
