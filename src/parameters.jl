abstract type AbstractParameter{T} end

"""
    value(x)

Return the "value" of an object.
For anything that is _not_ an `AbstractParameter` this is the identity function.
For `AbstractParameter`s this typically applies some transformation to some data
contained in the parameter, and returns a plain data type.
It might, for example, return a transformation of some internal data, the result of which
is guaranteed to satisfy some contraint.
"""
value(x) = x

"""
    Positive{T<:Real, V}

The `value` of a `Positive` is a `Real` number that is constrained to be positive. This is
represented in terms of an `unconstrained_value` and a `transform` that maps any value
the `unconstrained_value` might take to the positive reals.
"""
struct Positive{T<:Real, V<:Bijector} <: AbstractParameter{T}
    unconstrained_value::T
    transform::V
end

Positive(value::Real) = Positive(value, Bijectors.Exp())

value(x::Positive) = x.transform(x.unconstrained_value)

function flatten(x::Positive)
    v, unflatten_to_Real = flatten(x.unconstrained_value)

    function unflatten_to_Positive(v_new::Vector{<:Real})
        return Positive(unflatten_to_Real(v_new), x.transform)
    end

    return v, unflatten_to_Positive
end

"""
    Fixed{T}    

Represents a parameter whose value is required to stay constant. The `value` of a `Fixed` is
simply its value -- that constantness of the parameter is enforced by returning an empty
vector from `flatten`.
"""
struct Fixed{T} <: AbstractParameter{T}
    value::T
end

value(x::Fixed) = x.value

function flatten(x::Fixed)

    unflatten_to_Fixed(v_new::Vector{<:Real}) = x

    return Float64[], unflatten_to_Fixed
end
