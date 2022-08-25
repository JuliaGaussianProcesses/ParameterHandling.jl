@testset "parameters_array" begin
    @testset "postive" begin
        @testset "$val" for val in [[5.0, 4.0], [0.001f0], fill(1e-7, 1, 2)]
            p = positive(val)
            test_parameter_interface(p)
            @test value(p) ≈ val
            @test typeof(value(p)) === typeof(val)
        end

        # Test edge cases around the size of the value relative to the error tol.
        @test_throws ArgumentError positive([-0.1, 0.1])
        @test_throws ArgumentError positive(fill(1e-12, 1, 2, 3))
        @test value(positive(fill(1e-11, 3, 2, 1), exp, 1e-12)) ≈ fill(1e-11, 3, 2, 1)

        # Set a very loose bound on allocations, but one which is clearly sub-linear in
        # the size of `x`.
        @testset "zygote performance" begin
            x = rand(1000, 1000) .+ 0.1
            flat_x, unflatten = value_flatten(positive(x))

            # primal evaluation
            count_allocs(unflatten, flat_x)
            @test count_allocs(unflatten, flat_x) < 100

            # forward evaluation
            count_allocs(Zygote.pullback, unflatten, flat_x)
            @test count_allocs(Zygote.pullback, unflatten, flat_x) < 100

            # pullback
            out, pb = Zygote.pullback(unflatten, flat_x)
            count_allocs(pb, out)
            @test count_allocs(pb, out) < 100
        end
    end
end
