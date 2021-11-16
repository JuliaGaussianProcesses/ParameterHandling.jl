module TestUtils

using IterTools
using ParameterHandling
using Test

using ParameterHandling: AbstractParameter, value

# Handles equality of scalars, functions, tuples or arbitrary types
function default_equality(a::T, b::T; kwargs...) where {T}
    vals = fieldvalues(a)

    # If we don't have any fields then we're probably dealing with scalars
    if isempty(vals)
        # Only call isapprox for numbers, otherwise we fallback to ==
        return T <: Number ? isapprox(a, b; kwargs...) : a == b
    else
        return all(t -> default_equality(t...; kwargs...), zip(vals, fieldvalues(b)))
    end
end

# Handles extracting elements from arrays.
# Needed because fieldvalues(a) are empty, but we may need to recurse depending on
# the element type
function default_equality(a::T, b::T; kwargs...) where {T<:AbstractArray}
    return all(t -> default_equality(t...; kwargs...), zip(a, b))
end

# Handles extracting values for any dictionary types
function default_equality(a::T, b::T; kwargs...) where {T<:AbstractDict}
    return all(t -> default_equality(t...; kwargs...), zip(values(a), values(b)))
end

# NOTE: May want to make the equality function a kwarg in the future.
function test_flatten_interface(x::T; check_inferred::Bool=true) where {T}
    @testset "flatten($T)" begin
        # Checks default eltype still works and ensure that
        # basic functionality is implemented.
        v, unflatten = flatten(x)
        @test typeof(v) === Vector{Float64}
        @test default_equality(x, unflatten(v))
        @test unflatten(v) isa T

        # Check that everything infers properly.
        check_inferred && @inferred flatten(x)
        check_inferred && @inferred unflatten(v)

        # Test with different precisions
        @testset "Float64" begin
            _v, _unflatten = flatten(Float64, x)
            @test typeof(_v) === Vector{Float64}
            @test _v == v
            @test default_equality(x, unflatten(_v))
            @test _unflatten(_v) isa T

            # Check that everything infers properly.
            check_inferred && @inferred flatten(Float64, x)
            check_inferred && @inferred _unflatten(_v)
        end
        @testset "Float32" begin
            _v, _unflatten = flatten(Float32, x)
            @test typeof(_v) === Vector{Float32}
            @test default_equality(x, _unflatten(_v); atol=1e-5)
            @test _unflatten(_v) isa T

            # Check that everything infers properly.
            check_inferred && @inferred flatten(Float32, x)
            check_inferred && @inferred _unflatten(_v)
        end
        @testset "Float16" begin
            _v, _unflatten = flatten(Float16, x)
            @test typeof(_v) === Vector{Float16}
            @test default_equality(x, _unflatten(_v); atol=1e-2)
            @test _unflatten(_v) isa T

            # Check that everything infers properly.
            check_inferred && @inferred flatten(Float16, x)
            check_inferred && @inferred _unflatten(_v)
        end
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
