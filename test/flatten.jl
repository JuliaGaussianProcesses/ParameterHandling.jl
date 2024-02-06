@testset "flatten" begin
    @testset "Reals" begin
        test_flatten_interface(1.0)

        @testset "Integers" begin
            test_flatten_interface(1)
            @test isempty(first(flatten(1)))
        end
    end

    @testset "Nothing" begin
        test_flatten_interface(nothing)
        @test isempty(first(flatten(nothing)))
    end

    @testset "AbstractArrays" begin
        test_flatten_interface(randn(10))
        test_flatten_interface(randn(5, 4))
        test_flatten_interface([randn(5) for _ in 1:3])

        # Prevent regression of issue #31
        @testset for v in [[1, 2, 3], sparse([1, 0, 3])]
            test_flatten_interface(v)
            @test length(first(flatten(v))) == 0
        end
        # Prevent regression of issue #60
        @testset "Type stability of unflatten" begin
            θ = (rand(10),)
            x, unflatten = flatten(θ)
            @test (@inferred unflatten(x)) == θ
        end
    end

    @testset "SparseMatrixCSC" begin
        test_flatten_interface(sprand(10, 10, 0.5))
    end

    @testset "Tuple" begin
        test_flatten_interface((1.0, 2.0); check_inferred=tuple_infers)

        test_flatten_interface((1.0, (2.0, 3.0), randn(5)); check_inferred=tuple_infers)

        # Prevent regression of PR #67
        @testset "Type stability of unflatten" begin
            θ = (1.0, ((2.0, 3.0), 4.0))
            x, unflatten = flatten(θ)
            @test (@inferred unflatten(x)) == θ
        end
    end

    @testset "NamedTuple" begin
        test_flatten_interface(
            (a=1.0, b=(2.0, 3.0), c=(e=5.0,)); check_inferred=tuple_infers
        )
    end

    @testset "Dict" begin
        test_flatten_interface(Dict(:a => 4.0, :b => 5.0); check_inferred=false)
    end
end
