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

value(x::Fixed) = value(x.value)

function flatten(::Type{T}, x::Fixed) where {T<:Real}
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

struct Deferred{Tf,Targs} <: AbstractParameter
    f::Tf
    args::Targs
end

Base.:(==)(a::Deferred, b::Deferred) = (a.f == b.f) && (a.args == b.args)

value(x::Deferred) = x.f(value(x.args)...)

function flatten(::Type{T}, x::Deferred) where {T<:Real}
    v, unflatten = flatten(T, x.args)
    unflatten_Deferred(v_new::Vector{T}) = Deferred(x.f, unflatten(v_new))
    return v, unflatten_Deferred
end
