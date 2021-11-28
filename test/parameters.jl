@testset "parameters" begin
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
            θ -> only(Zygote.gradient(θ -> objective_function(unflatten, θ), θ)),
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
            θ -> only(Zygote.gradient(θ -> objective_function(unflatten, θ), θ)),
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
            θ -> only(Zygote.gradient(θ -> other_objective_function(unflatten, θ), θ)),
            flat_parameters,
            BFGS(),
            Optim.Options();
            inplace=false,
        )

        # Check that it's successfully optimised.
        @test mean(value(unflatten(results.minimizer))) ≈ 0 atol = 1e-7
    end

    @testset "value_flatten" begin
        x = (ones(3), fixed(5.0), (a=fixed(5.0), b=[6.0, 2.1]))
        v, unflatten = value_flatten(x)

        @test length(v) == 5
        @test unflatten(v) == ParameterHandling.value(x)
    end
end
