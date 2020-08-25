abstract type AbstractParameter end

"""
    value(x)

Return the "value" of an object.
For `AbstractParameter`s this typically applies some transformation to some data
contained in the parameter, and returns a plain data type.
It might, for example, return a transformation of some internal data, the result of which
is guaranteed to satisfy some contraint.
"""
value(x)

# Various basic `value` definitions.
value(x::Number) = x
value(x::AbstractArray{<:Number}) = x
value(x::AbstractArray) = map(value, x)
value(x::Tuple) = map(value, x)
value(x::NamedTuple) = map(value, x)
value(x::Dict) = Dict(k => value(v) for (k, v) in x)

"""
    Positive{T<:Real, V}

The `value` of a `Positive` is a `Real` number that is constrained to be positive. This is
represented in terms of an `unconstrained_value` and a `transform` that maps any value
the `unconstrained_value` might take to the positive reals.
"""
struct Positive{T<:Real, V<:Bijector} <: AbstractParameter
    unconstrained_value::T
    transform::V
end

Positive(value::Real) = Positive(value, Bijectors.Exp())

value(x::Positive) = x.transform(x.unconstrained_value)

function flatten(x::Positive)
    v, unflatten_to_Real = flatten(x.unconstrained_value)

    function unflatten_Positive(v_new::Vector{<:Real})
        return Positive(unflatten_to_Real(v_new), x.transform)
    end

    return v, unflatten_Positive
end

"""
    Fixed{T}    

Represents a parameter whose value is required to stay constant. The `value` of a `Fixed` is
simply its value -- that constantness of the parameter is enforced by returning an empty
vector from `flatten`.
"""
struct Fixed{T} <: AbstractParameter
    value::T
end

value(x::Fixed) = x.value

function flatten(x::Fixed)

    unflatten_Fixed(v_new::Vector{<:Real}) = x

    return Float64[], unflatten_Fixed
end

"""
    Deferred(f, args...)

The `value` of a `Deferred` is `f(value(args)...)`. This makes it possible to make the value
of the `args` e.g. `AbstractParameter`s and, therefore, enforce constraints on them even if
`f` knows nothing about `AbstractParameters`.
"""
struct Deferred{Tf, Targs} <: AbstractParameter
    f::Tf
    args::Targs
    function Deferred(f::Tf, args...) where {Tf}
        return new{Tf, typeof(args)}(f, args)
    end
end

value(x::Deferred) = x.f(map(value, x.args)...)

function flatten(x::Deferred)

    v, unflatten = flatten(x.args)

    function unflatten_Deferred(v_new::Vector{<:Real})
        return Deferred(x.f, unflatten(v_new)...)
    end

    return v, unflatten_Deferred
end
