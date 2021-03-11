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
    positive(val::T, transform::Bijector=Bijectors.Exp(), ε=sqrt(eps(T))) where T<:Real

Returns a `Postive`.
The `value` of a `Positive` is a `Real` number that is constrained to be positive.
This is represented in terms of an a `transform` that maps an `unconstrained_value` to the
positive reals.
Satisfies `val ≈ transform(unconstrained_value)`
"""
function positive(
    val::T, transform::Bijector=Bijectors.Exp(), ε=sqrt(eps(T)),
) where T<:Real
    val > 0 || throw(ArgumentError("Value ($val) is not positive."))
    val > ε || throw(ArgumentError("Value ($val) is too small, relative to ε ($ε)."))
    unconstrained_value = inv(transform)(val - ε)
    return Positive(unconstrained_value, transform, convert(typeof(unconstrained_value), ε))
end

struct Positive{T<:Real, V<:Bijector, Tε<:Real} <: AbstractParameter
    unconstrained_value::T
    transform::V
    ε::Tε
end

value(x::Positive) = x.transform(x.unconstrained_value) + x.ε

function flatten(::Type{T}, x::Positive) where T<:Real
    v, unflatten_to_Real = flatten(T, x.unconstrained_value)

    function unflatten_Positive(v_new::Vector{T})
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

function flatten(::Type{T}, x::Bounded) where T<:Real
    v, unflatten_to_Real = flatten(T, x.unconstrained_value)

    function unflatten_Bounded(v_new::Vector{T})
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

function flatten(::Type{T}, x::Fixed) where T<:Real
    unflatten_Fixed(v_new::Vector{T}) = x
    return T[], unflatten_Fixed
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

function flatten(::Type{T}, x::Deferred) where T<:Real
    v, unflatten = flatten(T, x.args)
    unflatten_Deferred(v_new::Vector{T}) = Deferred(x.f, unflatten(v_new))
    return v, unflatten_Deferred
end

"""
    nearest_orthogonal_matrix(X::StridedMatrix)

Project `X` onto the closest orthogonal matrix in Frobenius norm.

Originally used in varz: https://github.com/wesselb/varz/blob/master/varz/vars.py#L446
"""
@inline function nearest_orthogonal_matrix(X::StridedMatrix)
    # Inlining necessary for type inference for some reason.
    U, _, V = svd(X)
    return U * V'
end

"""
    orthogonal(X::StridedMatrix{<:Real})

Produce a parameter whose `value` is constrained to be positive. The argument `X` need not
be orthogonal.

This functionality projects `X` onto the nearest element subspace of orthogonal matrices (in
Frobenius norm) and is overparametrised as a consequence.

Originally used in varz: https://github.com/wesselb/varz/blob/master/varz/vars.py#L446
"""
orthogonal(X::StridedMatrix{<:Real}) = Orthogonal(X)

struct Orthogonal{TX<:StridedMatrix{<:Real}} <: AbstractParameter
    X::TX
end

Base.:(==)(X::Orthogonal, Y::Orthogonal) = X.X == Y.X

value(X::Orthogonal) = nearest_orthogonal_matrix(X.X)

function flatten(::Type{T}, X::Orthogonal) where {T<:Real}
    v, _unflatten = flatten(T, X.X)
    unflatten_Orthogonal(v_new::Vector{T}) = Orthogonal(_unflatten(v_new))
    return v, unflatten_Orthogonal
end
