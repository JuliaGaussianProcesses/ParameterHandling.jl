# Dummy type for use in testing.

struct Foo end

Base.zero(::Type{Foo}) = Foo()

function ParameterHandling.flatten(T, ::Foo)
    unflatten_Foo(x) = Foo()
    return T[], unflatten_Foo
end

LinearAlgebra.symmetric_type(::Type{Foo}) = Foo
LinearAlgebra.symmetric_type(::Type{Matrix{Foo}}) = Symmetric{Foo, Matrix{Foo}}

LinearAlgebra.symmetric(f::Foo, ::Symbol) = f

LinearAlgebra.transpose(f::Foo) = f
