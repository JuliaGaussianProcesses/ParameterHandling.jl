@testset "flatten" begin

    @testset "Reals" begin
        test_flatten_interface(1.0)

        @testset "Integers" begin
            test_flatten_interface(1)
            @test isempty(first(flatten(1)))
        end
    end

    @testset "AbstractArrays" begin
        # We round the rand output to avoid any precision issues when testing the flatten
        # interface (i.e., x::Float64 -> y::Float16 -> z::Float64 may result in x != z).
        # For testing purposes we don't care about this precision loss and we just want
        # make sure folks can use various precisions depending on the applications.
        test_flatten_interface(round.(randn(10); digits=3))
        test_flatten_interface(round.(randn(5, 4); digits=3))
        test_flatten_interface([round.(randn(5); digits=3) for _ in 1:3])
    end


    @testset "Tuple" begin

        test_flatten_interface((1.0, 2.0); check_inferred=tuple_infers)

        test_flatten_interface(
            (1.0, (2.0, 3.0), round.(randn(5); digits=3)); check_inferred=tuple_infers,
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
