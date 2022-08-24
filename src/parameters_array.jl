struct PositiveArray{T<:Array{<:Real},V,Tε<:Real} <: AbstractParameter
    unconstrained_value::T
    transform::V
    ε::Tε
end

ParameterHandling.value(x::PositiveArray) = map(exp, x.x) .+ 1e-4

function ParameterHandling.flatten(::Type{T}, x::PositiveArray{<:Array{T}}) where {T<:Real}
    v, unflatten_to_array = flatten(x.unconstrained_value)
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
