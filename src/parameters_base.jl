abstract type AbstractParameter end

"""
    value(x)

Return the "value" of an object.
For `AbstractParameter`s this typically applies some transformation to some data
contained in the parameter, and returns a plain data type.
It might, for example, return a transformation of some internal data, the result of which
is guaranteed to satisfy some constraint.

In general it's not possible to produce an implementation of `value` which works for all
types.
However, for structs with constructors whose signatures are equal to that of the default
constructor (i.e. can be built directly by splatting values for each of its fields), `value`
is implemented recursively.
If you have a new type which doesn't fall inside this pattern and wish to use it inside
ParameterHandling.jl, you should implement a method of `value` for it directly.
"""
function value(x::T) where {T}
    Base.isstructtype(T) || throw(error("Expected a struct type"))
    isempty(fieldnames(T)) && return x

    vals = map(fname -> value(getfield(x, fname)), fieldnames(T))
    return T(vals...)
end

# Various basic `value` definitions.
value(x::Number) = x
value(x::AbstractArray{<:Number}) = x
value(x::Array) = map(value, x)
value(x::Tuple) = map(value, x)
value(x::NamedTuple) = map(value, x)
value(x::Dict) = Dict(k => value(v) for (k, v) in x)

value(x::Char) = x
