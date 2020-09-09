using ParameterHandling: Positive, Bounded

@testset "parameters" begin

    @testset "postive" begin
        @testset "$val" for val in [5.0, 1e-11, 1e-12]
            p = positive(val)
            test_parameter_interface(p)
            @test value(p) ≈ val
        end
    end

    @testset "bounded" begin
        @testset "$val" for val in [-0.05, -0.1 + 1e-12, 2.0 - 1e-11, 2.0 - 1e-12]
            p = bounded(val, -0.1, 2.0)
            test_parameter_interface(bounded(-0.05, -0.1, 2.0))
            @test value(p) ≈ val
        end
    end

    @testset "fixed" begin
        val = (a=5.0, b=4.0)
        p = fixed(val)
        test_parameter_interface(p)
        @test value(p) == val
    end

    @testset "deferred" begin
        test_parameter_interface(deferred(sin, 0.5))
        test_parameter_interface(deferred(sin, positive(0.5)))
        test_parameter_interface(
            deferred(
                MvNormal,
                fixed(randn(5)),
                deferred(PDiagMat, positive.(rand(5) .+ 1e-1)),
            )
        )
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
