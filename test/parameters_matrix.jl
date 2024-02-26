using ParameterHandling: vec_to_tril, tril_to_vec

@testset "parameters_matrix.jl" begin
    @testset "orthogonal" begin
        is_almost_orthogonal(X::AbstractMatrix, tol) = norm(X'X - I) < tol

        @testset "nearest_orthogonal_matrix($T)" for T in [Float64, ComplexF64]
            X_orth = ParameterHandling.nearest_orthogonal_matrix(randn(T, 5, 4))
            @test is_almost_orthogonal(X_orth, 1e-9)
            X_orth_2 = ParameterHandling.nearest_orthogonal_matrix(X_orth)
            @test X_orth ≈ X_orth_2 # nearest_orthogonal_matrix is a projection.
        end

        X = orthogonal(randn(5, 4))
        @test X == X
        test_parameter_interface(X)
        @test is_almost_orthogonal(value(X), 1e-9)

        # We do not implement any custom rrules, so we only check that `Zygote` is able to
        # differentiate, and assume that the result is correct if it doesn't error.
        @testset "Zygote" begin
            _, pb = Zygote.pullback(X -> value(orthogonal(X)), randn(3, 2))
            @test only(pb(randn(3, 2))) isa Matrix{<:Real}
        end
    end

    @testset "positive_definite" begin
        @testset "vec_tril_conversion" begin
            X = tril!(rand(3, 3))
            @test vec_to_tril(tril_to_vec(X)) == X
            @test_throws ErrorException tril_to_vec(rand(4, 5))
        end
        X_mat = ParameterHandling.A_At(rand(3, 3)) # Create a positive definite object
        X = positive_definite(X_mat)
        @test value(X) ≈ X_mat
        @test isposdef(value(X))
        @test vec_to_tril(X.L) ≈ cholesky(X_mat).L

        # Zeroing the unconstrained value preserve positivity
        X.L .= 0 
        @test isposdef(value(X))

        # Check that zeroing the unconstrained value does not affect the original array
        @test X != positive_definite(X_mat)

        @test positive_definite(X_mat, 1e-5) != positive_definite(X_mat, 1e-6)
        @test_throws ArgumentError positive_definite(zeros(3, 3))
        @test_throws ArgumentError positive_definite(X_mat, 0.0)

        test_parameter_interface(X)

        x, re = flatten(X)
        Δl = first(
            Zygote.gradient(x) do x
                X = re(x)
                return logdet(value(X))
            end,
        )
        ChainRulesTestUtils.test_rrule(vec_to_tril, x)
    end
end
