module TestUtils

using IterTools
using ParameterHandling
using Test

using ParameterHandling: AbstractParameter, value

# Handles equality of structs / mutable structs.
function default_equality(a::Ta, b::Tb; kwargs...) where {Ta,Tb}
    (isstructtype(Ta) && isstructtype(Tb)) || throw(error("Arguments aren't structs"))
    return all(t -> default_equality(t...; kwargs...), zip(fieldvalues(a), fieldvalues(b)))
end

default_equality(a::Number, b::Number; kwargs...) = isapprox(a, b; kwargs...)

# Handles extracting elements from arrays.
# Needed because fieldvalues(a) are empty, but we may need to recurse depending on
# the element type
function default_equality(a::AbstractArray, b::AbstractArray; kwargs...)
    return all(t -> default_equality(t...; kwargs...), zip(a, b))
end

# Handles extracting values for any dictionary types
function default_equality(a::AbstractDict, b::AbstractDict; kwargs...)
    return all(t -> default_equality(t...; kwargs...), zip(values(a), values(b)))
end

struct MyReal{T} <: Real
    v::T
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

        # Test with different precisions
        @testset "Float64" begin
            _v, _unflatten = flatten(Float64, x)
            @test typeof(_v) === Vector{Float64}
            @test _v == v
            @test default_equality(x, unflatten(_v))

            # Check that unflattening works with different reals.
            _unflatten(map(MyReal, randn(length(_v))))

            # Check that everything infers properly.
            check_inferred && @inferred flatten(Float64, x)
        end
        @testset "Float32" begin
            _v, _unflatten = flatten(Float32, x)
            @test typeof(_v) === Vector{Float32}
            @test default_equality(x, _unflatten(_v); atol=1e-5)

            # Check that unflattening works with different precisions.
            _unflatten(map(MyReal, randn(length(_v))))

            # Check that everything infers properly.
            check_inferred && @inferred flatten(Float32, x)
        end
        @testset "Float16" begin
            _v, _unflatten = flatten(Float16, x)
            @test typeof(_v) === Vector{Float16}
            @test default_equality(x, _unflatten(_v); atol=1e-2)

            # Check that unflattening works with different precisions.
            _unflatten(map(MyReal, randn(length(_v))))

            # Check that everything infers properly.
            check_inferred && @inferred flatten(Float16, x)
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
