module TestUtils

using ParameterHandling
using Test

using ParameterHandling: AbstractParameter, value

function test_flatten_interface(x::T; check_inferred::Bool=true) where {T}

    # Ensure that basic functionality is implemented.
    v, unflatten = flatten(x)
    @test v isa Vector{<:Real}
    @test x == unflatten(v)
    @test unflatten(v) isa T

    # Check that everything infers properly.
    if check_inferred
        @inferred flatten(x)
    end

    return nothing
end

function test_parameter_interface(x; check_inferred::Bool=true)

    # Parameters need to be flatten-able.
    test_flatten_interface(x; check_inferred=check_inferred)

    # Run this to make sure that it doesn't error.
    value(x)

    if check_inferred
        @inferred value(x)
    end
    return nothing
end

end # module
