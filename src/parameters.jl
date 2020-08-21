abstract type AbstractParameter{T} end

value(x) = x

"""
    Positive{T<:Real, V}


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


"""
struct Fixed{T} <: AbstractParameter{T}
    value::T
end

value(x::Fixed) = x.value

function flatten(x::Fixed)
    v, unflatten = flatten(x.value)

    unflatten_to_Fixed(v_new::Vector{<:Real}) = x

    return v, unflatten_to_Fixed
end
