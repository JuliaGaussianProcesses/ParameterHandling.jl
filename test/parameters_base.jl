struct Foo end

Base.zero(::Type{Foo}) = Foo()

@testset "parameters_base" begin
    @test value(Diagonal([Foo() for _ in 1:5])) isa Diagonal
end
