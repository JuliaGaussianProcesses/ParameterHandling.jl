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

        # These tests assume that if the number of allocations is roughly constant in the
        # size of `x`, then performance is acceptable. This is demonstrated by requiring
        # that the number of allocations (100) is a lot smaller than the total length of
        # the array in question (1_000_000). The bound (100) is quite loose because there
        # are typically serveral 10s of allocations made by Zygote for book-keeping
        # purposes etc.
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

        # Check that this optimisation is actually necessary -- i.e. that the performance
        # of the equivalent operation, `map(positive, x)`, is indeed poor, esp. with AD.
        # Poor performance is demonstrated by showing that there's at least one allocation
        # per element. A smaller array than the previous test set is used because it can
        # be _really_ slow for large arrays (several seconds), which is undesirable in
        # unit tests.
        @testset "zygote performance of scalar equivalent" begin
            x = rand(1000) .+ 0.1
            flat_x, unflatten = value_flatten(map(positive, x))

            # forward evaluation
            count_allocs(Zygote.pullback, unflatten, flat_x)
            @test count_allocs(Zygote.pullback, unflatten, flat_x) > 1000
        end
    end

    @testset "bounded" begin
        @testset "$val" for val in [
            [-0.05, 0.5], [-0.1 + 1e-12, 2.0 - 1e-11], fill(2.0 - 1e-12, 1, 2, 3)
        ]
            p = bounded(val, -0.1, 2.0)
            test_parameter_interface(p)
            @test value(p) ≈ val
        end

        @test_throws ArgumentError bounded([-0.05], 0.0, 1.0)

        # Same style of performance test as for positive(::Array). See above for info.
        @testset "zygote performance" begin
            x = rand(1000, 1000) .* 1.98 .- 0.99
            flat_x, unflatten = value_flatten(bounded(x, -1.0, 1.0))

            # primal evaluation
            count_allocs(unflatten, flat_x)
            @test count_allocs(unflatten, flat_x) < 300

            # forward evaluation
            count_allocs(Zygote.pullback, unflatten, flat_x)
            @test count_allocs(Zygote.pullback, unflatten, flat_x) < 300

            # pullback
            out, pb = Zygote.pullback(unflatten, flat_x)
            count_allocs(pb, out)
            @test count_allocs(pb, out) < 300
        end

        # Same style of performance test as for `map(positive, x)`. See above for info.
        @testset "zygote performance of scalar equivalent" begin
            x = rand(1000) .* 1.98 .- 0.99
            flat_x, unflatten = value_flatten(map(x -> bounded(x, -1.0, 1.0), x))

            # forward evaluation
            count_allocs(Zygote.pullback, unflatten, flat_x)
            @test count_allocs(Zygote.pullback, unflatten, flat_x) > 1000
        end
    end
end
