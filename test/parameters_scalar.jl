using ParameterHandling: Positive, Bounded

@testset "parameters_scalar.jl" begin
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
        @test value(positive(1e-11, exp, 1e-12)) ≈ 1e-11
    end

    @testset "bounded" begin
        @testset "$val" for val in [-0.05, -0.1 + 1e-12, 2.0 - 1e-11, 2.0 - 1e-12]
            p = bounded(val, -0.1, 2.0)
            test_parameter_interface(p)
            @test value(p) ≈ val
        end

        @test_throws ArgumentError bounded(-0.05, 0.0, 1.0)
    end
end
