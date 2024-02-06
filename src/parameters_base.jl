abstract type AbstractParameter end

"""
    value(x)

Return the "value" of an object.
For `AbstractParameter`s this typically applies some transformation to some data
contained in the parameter, and returns a plain data type.
It might, for example, return a transformation of some internal data, the result of which
is guaranteed to satisfy some constraint.
"""
value(x)

# Various basic `value` definitions.
value(x::Number) = x
value(x::AbstractArray{<:Number}) = x
value(x::AbstractArray) = map(value, x)
value(x::Tuple) = map(value, x)
value(x::NamedTuple) = map(value, x)
value(x::Dict) = Dict(k => value(v) for (k, v) in x)
value(::Nothing) = nothing
