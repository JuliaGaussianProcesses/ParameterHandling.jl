@testset "flatten" begin

    @testset "Reals" begin
        test_flatten_interface(1.0)

        @testset "Integers" begin
            test_flatten_interface(1)
            @test isempty(first(flatten(1)))
        end
    end

    @testset "AbstractArrays" begin
        test_flatten_interface(randn(10))
        test_flatten_interface(randn(5, 4))
        test_flatten_interface([randn(5) for _ in 1:3])
    end

    @testset "Tuple" begin
        test_flatten_interface((1.0, 2.0); check_inferred=tuple_infers)

        test_flatten_interface(
            (1.0, (2.0, 3.0), randn(5)); check_inferred=tuple_infers,
        )
    end

    @testset "NamedTuple" begin
        test_flatten_interface(
            (a=1.0, b=(2.0, 3.0), c=(e=5.0,)); check_inferred=tuple_infers,
        )
    end

    @testset "Dict" begin
        test_flatten_interface(Dict(:a => 4.0, :b => 5.0); check_inferred=false)
    end
end
