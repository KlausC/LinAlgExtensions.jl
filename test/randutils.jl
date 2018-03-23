

srand(1)

let
@testset "orthogonality $((m, n))" for (m, n) in ((5, 5), (10, 4), (3, 7)) 
    X = randorth(m, n)
    @test isorth(X)
end


@testset "singular values $((m, n))" for (m, n) in ((5, 5), (10, 4), (3, 7)) 
    sv = [0.01; 1.0; 0.1]
    k = min(m, n)
    X = randrealsv(sv, m, n)
    U, S, V = svd(X)
    @test S ≈ sort([sv; zeros(k-length(sv))], rev=true)
    @test size(U) == (m, k)
    @test size(V) == (n, k)
end

@testset "sparse matrices $((m, n, k))" for (m, n, k) in ((10, 5, 3), (5, 10, 4))
    X = randsparse(m, n, k, 0.3)
    @test count(diag(qrfact(X).R) .!= 0) == k
end


end # let

