module TestUtils

using IterTools
using ParameterHandling
using Test

using ParameterHandling: AbstractParameter, value

# Handles equality of scalars, functions, tuples or arbitrary types
function default_equality(a::T, b::T) where T
    vals = fieldvalues(a)

    # If we don't have
    if isempty(vals)
        # Only call isapprox for numbers, otherwise we fallback to ==
        return T <: Number ? isapprox(a, b; atol=1e-2) : a == b
    else
        return all(t -> default_equality(t...), zip(vals, fieldvalues(b)))
    end
end

# Handles extracting elements from arrays.
# Needed because fieldvalues(a) are empty, but we may need to recurse depending on
# the element type
function default_equality(a::T, b::T) where T<:AbstractArray
    return all(t -> default_equality(t...), zip(a, b))
end

# Handles extracting values for any dictionary types
function default_equality(a::T, b::T) where T<:AbstractDict
    return all(t -> default_equality(t...), zip(values(a), values(b)))
end

function test_flatten_interface(
    x::T;
    check_inferred::Bool=true,
    eltypes=(Float16, Float32, Float64),
    equality=default_equality,
) where {T}

    # Checks default eltype still works
    v, unflatten = flatten(x)
    @test typeof(v) === Vector{Float64}
    @test equality(x, unflatten(v))
    @test unflatten(v) isa T

    # Check that everything infers properly.
    check_inferred && @inferred flatten(x)

    # Test with different precisions
    @testset "flatten($type, $T)" for type in eltypes
        # Ensure that basic functionality is implemented.
        v, unflatten = flatten(type, x)
        @test typeof(v) === Vector{type}
        @test equality(x, unflatten(v))
        @test unflatten(v) isa T

        # Check that everything infers properly.
        check_inferred && @inferred flatten(type, x)
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
