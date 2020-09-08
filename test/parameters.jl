@testset "parameters" begin

    @testset "Postive" begin
        test_parameter_interface(Positive(5.0))
        p = Positive(-10.0)
        @test value(p) > p.ε
        p = Positive(-Inf)
        @test value(p) == p.ε
    end

    @testset "Bounded" begin
        test_parameter_interface(Bounded(-1.0, -0.1, 2.0))
        p = Bounded(-10.0, -0.1, 2.0)
        @test value(p) > p.lower_bound
        p = Bounded(-Inf, -0.1, 2.0)
        @test value(p) == p.lower_bound + p.ε
        p = Bounded(10.0, -0.1, 2.0)
        @test value(p) < p.upper_bound
        p = Bounded(Inf, -0.1, 2.0)
        @test value(p) == p.upper_bound - p.ε
    end

    @testset "Fixed" begin
        test_parameter_interface(Fixed((a=5.0, b=4.0)))
    end

    @testset "Deferred" begin
        test_parameter_interface(Deferred(sin, 0.5))
        test_parameter_interface(Deferred(sin, Positive(log(0.5))))
        test_parameter_interface(
            Deferred(
                MvNormal,
                Fixed(randn(5)),
                Deferred(PDiagMat, Positive.(randn(5))),
            )
        )
    end

    function objective_function(unflatten, flat_θ::Vector{<:Real})
        θ = value(unflatten(flat_θ))
        return abs2(θ.a) + abs2(θ.b)
    end

    # This is more of a worked example. Will be properly split up / tidied up.
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

        θ0 = (a=5.0, b=Fixed(4.0))
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

        θ0 = Deferred(Normal, randn(), Positive(log(1.0)))
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
