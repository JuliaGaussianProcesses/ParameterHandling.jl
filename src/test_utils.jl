module TestUtils

using ParameterHandling
using Test

using ParameterHandling: AbstractParameter, value

function test_flatten_interface(x::T) where {T}
    v, unflatten = flatten(x)
    @test v isa Vector{<:Real}
    @test x == unflatten(v)
    @test unflatten(v) isa T
    return nothing
end

function test_parameter_interface(x::AbstractParameter)

    # Parameters need to be flatten-able.
    test_flatten_interface(x)

    # Run this to make sure that it doesn't error.
    value(x)
    return nothing
end

end # module
