using ParameterHandling: Positive, Bounded
using ParameterHandling: vec_to_tril, tril_to_vec

mvnormal(args...) = MvNormal(args...)
pdiagmat(args...) = PDiagMat(args...)

@testset "parameters" begin

    @testset "postive" begin
        @testset "$val" for val in [5.0, 0.001f0, 1.0e-7]
            p = positive(val)
            test_parameter_interface(p)
            @test value(p) ≈ val
            @test typeof(value(p)) === typeof(val)
        end

        # Test edge cases around the size of the value relative to the error tol.
        @test_throws ArgumentError positive(-0.1)
        @test_throws ArgumentError positive(1e-11)
        @test value(positive(1e-11, Bijectors.Exp(), 1e-12)) ≈ 1e-11
    end

    @testset "bounded" begin
        @testset "$val" for val in [-0.05, -0.1 + 1e-12, 2.0 - 1e-11, 2.0 - 1e-12]
            p = bounded(val, -0.1, 2.0)
            test_parameter_interface(p)
            @test value(p) ≈ val
        end

        @test_throws ArgumentError bounded(-0.05, 0.0, 1.0)
    end

    @testset "fixed" begin
        val = (a=5.0, b=4.0)
        p = fixed(val)
        test_parameter_interface(p)
        @test value(p) == val
    end

    @testset "deferred" begin
        test_parameter_interface(deferred(sin, 0.5); check_inferred=tuple_infers)
        test_parameter_interface(deferred(sin, positive(0.5)); check_inferred=tuple_infers)
        test_parameter_interface(
            deferred(
                mvnormal,
                fixed(randn(5)),
                deferred(pdiagmat, positive.(rand(5) .+ 1e-1)),
            );
            check_inferred=tuple_infers,
        )
    end

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
        @test X == X
        @test value(X) ≈ X_mat
        @test isposdef(value(X))
        @test vec_to_tril(X.L) ≈ cholesky(X_mat).L
        @test_throws ArgumentError positive_definite(rand(3, 3))
        test_parameter_interface(X)
    end

    function objective_function(unflatten, flat_θ::Vector{<:Real})
        θ = value(unflatten(flat_θ))
        return abs2(θ.a) + abs2(θ.b)
    end

    # This is more of a worked example.
    @testset "Integration" begin

        θ0 = (a=5.0, b=4.0)
        flat_parameters, unflatten = flatten(θ0)

        results = Optim.optimize(
            θ -> objective_function(unflatten, θ),
            θ->only(Zygote.gradient(θ -> objective_function(unflatten, θ), θ)),
            flat_parameters,
            BFGS(),
            Optim.Options();
            inplace=false,
        )

        # Check that it's successfully optimised.
        @test objective_function(unflatten, results.minimizer) < 1e-12
    end

    @testset "Other Integration" begin

        θ0 = (a=5.0, b=fixed(4.0))
        flat_parameters, unflatten = flatten(θ0)

        results = Optim.optimize(
            θ -> objective_function(unflatten, θ),
            θ->only(Zygote.gradient(θ -> objective_function(unflatten, θ), θ)),
            flat_parameters,
            BFGS(),
            Optim.Options();
            inplace=false,
        )

        # Check that it's successfully optimised.
        @test value(unflatten(results.minimizer).b) == 4.0
    end

    function other_objective_function(unflatten, flat_θ::Vector{<:Real})
        X = value(unflatten(flat_θ))
        return -logpdf(X, -1.0) - logpdf(X, 1.0)
    end

    @testset "Normal" begin

        θ0 = deferred(Normal, randn(), positive(1.0))
        flat_parameters, unflatten = flatten(θ0)

        results = Optim.optimize(
            θ -> other_objective_function(unflatten, θ),
            θ->only(Zygote.gradient(θ -> other_objective_function(unflatten, θ), θ)),
            flat_parameters,
            BFGS(),
            Optim.Options();
            inplace=false,
        )

        # Check that it's successfully optimised.
        @test mean(value(unflatten(results.minimizer))) ≈ 0 atol=1e-7
    end
end
