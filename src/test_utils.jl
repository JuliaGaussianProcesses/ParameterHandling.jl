module TestUtils

using ParameterHandling
using Test

function test_flatten_interface(x::T) where {T}
    v, unflatten = flatten(x)
    @test v isa Vector{<:Real}
    @test x == unflatten(v)
    @test unflatten(v) isa T
    return nothing
end

end # module
