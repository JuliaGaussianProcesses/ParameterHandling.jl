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
    positive(val::Real, transform::Bijector=Bijectors.Exp(), ε::Real = 1e-12)

Returns a `Postive`.
The `value` of a `Positive` is a `Real` number that is constrained to be positive.
This is represented in terms of an a `transform` that maps an `unconstrained_value` to the
positive reals.
Satisfies `val ≈ transform(unconstrained_value)`
"""
function positive(val::Real, transform::Bijector=Bijectors.Exp(), ε::Real = 1e-12)
    if val <= 0
        throw(ArgumentError("Value, $val, is not positive."))
    end
    unconstrained_value = inv(transform)(val - ε)
    return Positive(unconstrained_value, transform, convert(typeof(unconstrained_value), ε))
end

struct Positive{T<:Real, V<:Bijector, Tε<:Real} <: AbstractParameter
    unconstrained_value::T
    transform::V
    ε::Tε
end

value(x::Positive) = x.transform(x.unconstrained_value) + x.ε

function flatten(x::Positive)
    v, unflatten_to_Real = flatten(x.unconstrained_value)

    function unflatten_Positive(v_new::Vector{<:Real})
        return Positive(unflatten_to_Real(v_new), x.transform, x.ε)
    end

    return v, unflatten_Positive
end

"""
    bounded(val::Real, lower_bound::Real, upper_bound::Real)

Constructs a `Bounded`.
The `value` of a `Bounded` is a `Real` number that is constrained to be within the interval
(`lower_bound`, `upper_bound`), and is equal to `val`.
This is represented internally in terms of an `unconstrained_value` and a `transform` that
maps any real to this interval. `unconstrained_value` is `inv(transform)(val)`.
"""
function bounded(val::Real, lower_bound::Real, upper_bound::Real)
    lb = convert(typeof(val), lower_bound)
    ub = convert(typeof(val), upper_bound)
    ε = convert(typeof(val), 1e-12)

    if val > upper_bound || val < lower_bound
        throw(ArgumentError(
            "Value, $val, outside of specified bounds ($lower_bound, $upper_bound).",
        ))
    end

    inv_transform = Bijectors.Logit(lb + ε, ub - ε)
    transform = inv(inv_transform)

    # Bijectors defines only Logit struct so we use Logistic as the inverse of Logit
    return Bounded(inv_transform(val), lb, ub, transform, ε)
end

struct Bounded{T<:Real, V<:Bijector, Tε<:Real} <: AbstractParameter
    unconstrained_value::T
    lower_bound::T
    upper_bound::T
    transform::V
    ε::Tε
end

value(x::Bounded) = x.transform(x.unconstrained_value)

function flatten(x::Bounded)
    v, unflatten_to_Real = flatten(x.unconstrained_value)

    function unflatten_Bounded(v_new::Vector{<:Real})
        return Bounded(
            unflatten_to_Real(v_new), x.lower_bound, x.upper_bound, x.transform, x.ε,
        )
    end

    return v, unflatten_Bounded
end

"""
    fixed(val)

Represents a parameter whose value is required to stay constant. The `value` of a `Fixed` is
simply `val`. Constantness of the parameter is enforced by returning an empty
vector from `flatten`.
"""
fixed(val) = Fixed(val)

struct Fixed{T} <: AbstractParameter
    value::T
end

value(x::Fixed) = x.value

function flatten(x::Fixed)

    unflatten_Fixed(v_new::Vector{<:Real}) = x

    return Float64[], unflatten_Fixed
end

"""
    deferred(f, args...)

The `value` of a `deferred` is `f(value(args)...)`. This makes it possible to make the value
of the `args` e.g. `AbstractParameter`s and, therefore, enforce constraints on them even if
`f` knows nothing about `AbstractParameters`.

It can be helpful to use `deferred` recursively when constructing complicated objects.
"""
deferred(f, args...) = Deferred(f, args)

struct Deferred{Tf, Targs} <: AbstractParameter
    f::Tf
    args::Targs
end

Base.:(==)(a::Deferred, b::Deferred) = (a.f == b.f) && (a.args == b.args)

value(x::Deferred) = x.f(value(x.args)...)

function flatten(x::Deferred)

    v, unflatten = flatten(x.args)

    function unflatten_Deferred(v_new::Vector{<:Real})
        return Deferred(x.f, unflatten(v_new))
    end

    return v, unflatten_Deferred
end
