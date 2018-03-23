
srand(1)

let
    @testset "pinvfactors $((m, n, k))" for (m, n, k) in ((100, 50, 40), (50, 100, 40))
        X = randsparse(m, n, k, 1000/m/n)
        XM = Matrix(X)
        XMI = pinv(XM)
        pin = pinvfact(X)
        @test rank(pin) == k
        @test pin \ Matrix(1.0I, m, m) ≈ XMI
        @test pin' \ Matrix(1.0I, n, n) ≈ XMI'
        pin = pinvfact(XM)
        @test rank(pin) == k
        @test pin \ Matrix(1.0I, m, m) ≈ XMI
        @test pin' \ Matrix(1.0I, n, n) ≈ XMI'
    end
end
