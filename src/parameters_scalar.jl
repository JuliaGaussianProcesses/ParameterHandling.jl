"""
    positive(val::Real, transform=exp, ε=sqrt(eps(typeof(val))))

Return a `Positive`.
The `value` of a `Positive` is a `Real` number that is constrained to be positive.
This is represented in terms of a `transform` that maps an `unconstrained_value` to the
positive reals.
Satisfies `val ≈ transform(unconstrained_value)`.
"""
function positive(val::Real, transform=exp, ε=sqrt(eps(typeof(val))))
    val > 0 || throw(ArgumentError("Value ($val) is not positive."))
    val > ε || throw(ArgumentError("Value ($val) is too small, relative to ε ($ε)."))
    unconstrained_value = inverse(transform)(val - ε)
    return Positive(unconstrained_value, transform, convert(typeof(unconstrained_value), ε))
end

struct Positive{T<:Real,V,Tε<:Real} <: AbstractParameter
    unconstrained_value::T
    transform::V
    ε::Tε
end

value(x::Positive) = x.transform(x.unconstrained_value) + x.ε

function flatten(::Type{T}, x::Positive) where {T<:Real}
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
maps any `Real` to this interval.
"""
function bounded(val::Real, lower_bound::Real, upper_bound::Real)
    lb = convert(typeof(val), lower_bound)
    ub = convert(typeof(val), upper_bound)

    # construct open interval
    ε = convert(typeof(val), 1e-12)
    lb_plus_ε = lb + ε
    ub_minus_ε = ub - ε

    if val > ub_minus_ε || val < lb_plus_ε
        throw(
            ArgumentError(
                "Value, $val, outside of specified bounds ($lower_bound, $upper_bound)."
            ),
        )
    end

    length_interval = ub_minus_ε - lb_plus_ε
    unconstrained_val = logit((val - lb_plus_ε) / length_interval)
    transform(x) = lb_plus_ε + length_interval * logistic(x)

    return Bounded(unconstrained_val, lb, ub, transform, ε)
end

struct Bounded{T<:Real,V,Tε<:Real} <: AbstractParameter
    unconstrained_value::T
    lower_bound::T
    upper_bound::T
    transform::V
    ε::Tε
end

value(x::Bounded) = x.transform(x.unconstrained_value)

function flatten(::Type{T}, x::Bounded) where {T<:Real}
    v, unflatten_to_Real = flatten(T, x.unconstrained_value)

    function unflatten_Bounded(v_new::Vector{T})
        return Bounded(
            unflatten_to_Real(v_new), x.lower_bound, x.upper_bound, x.transform, x.ε
        )
    end

    return v, unflatten_Bounded
end
