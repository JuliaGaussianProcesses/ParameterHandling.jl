struct PositiveArray{T<:Array{<:Real},V,Tε<:Real} <: AbstractParameter
    unconstrained_value::T
    transform::V
    ε::Tε
end

value(x::PositiveArray) = map(exp, x.unconstrained_value) .+ x.ε

function flatten(::Type{T}, x::PositiveArray{<:Array{V}}) where {T<:Real,V<:Real}
    v, unflatten_to_array = flatten(T, x.unconstrained_value)
    transform = x.transform
    ε = x.ε
    function unflatten_PositiveArray(v::AbstractVector{T})
        return PositiveArray(unflatten_to_array(v), transform, ε)
    end
    return v, unflatten_PositiveArray
end

"""
    positive(x::Array{<:Real})

Roughly equivalent to `map(positive, x)`, but implemented such that unflattening can be
efficiently differentiated through using algorithmic differentiation (Zygote in particular).
"""
function positive(val::Array{<:Real}, transform=exp, ε=sqrt(eps(eltype(val))))
    all(val .> 0) || throw(ArgumentError("Not all elements of val are positive."))
    all(val .> ε) || throw(ArgumentError("Not all elements of val greater than ε ($ε)."))

    inverse_transform = inverse(transform)
    unconstrained_value = map(x -> inverse_transform(x - ε), val)
    return PositiveArray(
        unconstrained_value, transform, convert(eltype(unconstrained_value), ε)
    )
end

struct BoundedArray{T<:Real,Ta<:AbstractArray{T},V,Tε<:Real} <: AbstractParameter
    unconstrained_value::Ta
    lower_bound::T
    upper_bound::T
    transform::V
    ε::Tε
end

value(x::BoundedArray) = x.transform(x.unconstrained_value)

function flatten(::Type{T}, x::BoundedArray) where {T<:Real}
    v, unflatten_to_Array = flatten(T, x.unconstrained_value)

    function unflatten_Bounded(v_new::Vector{T})
        return BoundedArray(
            unflatten_to_Array(v_new), x.lower_bound, x.upper_bound, x.transform, x.ε
        )
    end

    return v, unflatten_Bounded
end

"""
    bounded(val::Array{<:Real}, lower_bound::Real, upper_bound::Real)

Roughly equivalent to `bounded.(val, lower_bound, upper_bound)`, but implemented such that
unflattening can be efficiently differentiated through using algorithmic differentiation
(Zygote in particular).
"""
function bounded(val::Array{<:Real}, lower_bound::Real, upper_bound::Real)
    lb = convert(eltype(val), lower_bound)
    ub = convert(eltype(val), upper_bound)

    # construct open interval
    ε = convert(eltype(val), 1e-12)
    lb_plus_ε = lb + ε
    ub_minus_ε = ub - ε

    if any(val .> ub_minus_ε) || any(val .< lb_plus_ε)
        throw(
            ArgumentError("At least one element of `val`, $val, outside of specified bounds
                          ($lower_bound, $upper_bound).")
        )
    end

    length_interval = ub_minus_ε - lb_plus_ε
    unconstrained_val = logit.((val .- lb_plus_ε) ./ length_interval)
    transform(x) = lb_plus_ε .+ length_interval .* logistic.(x)

    return BoundedArray(unconstrained_val, lb, ub, transform, ε)
end
