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

"""
    nearest_orthogonal_matrix(X::StridedMatrix)

Project `X` onto the closest orthogonal matrix in Frobenius norm.

Originally used in varz: https://github.com/wesselb/varz/blob/master/varz/vars.py#L446
"""
@inline function nearest_orthogonal_matrix(X::StridedMatrix{<:Union{Real,Complex}})
    # Inlining necessary for type inference for some reason.
    U, _, V = svd(X)
    return U * V'
end

"""
    orthogonal(X::StridedMatrix{<:Real})

Produce a parameter whose `value` is constrained to be an orthogonal matrix. The argument `X` need not
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
    v, unflatten_to_Array = flatten(T, X.X)
    unflatten_Orthogonal(v_new::Vector{T}) = Orthogonal(unflatten_to_Array(v_new))
    return v, unflatten_Orthogonal
end

"""
    positive_definite(X::StridedMatrix{<:Real})

Produce a parameter whose `value` is constrained to be a positive-definite matrix. The argument `X` needs to
be a positive-definite matrix (see https://en.wikipedia.org/wiki/Definite_matrix).

The unconstrained parameter is a `LowerTriangular` matrix, stored as a vector.
"""
function positive_definite(X::StridedMatrix{<:Real})
    isposdef(X) || throw(ArgumentError("X is not positive-definite"))
    return PositiveDefinite(tril_to_vec(cholesky(X).L))
end

struct PositiveDefinite{TL<:AbstractVector{<:Real}} <: AbstractParameter
    L::TL
end

Base.:(==)(X::PositiveDefinite, Y::PositiveDefinite) = X.L == Y.L

A_At(X) = X * X'

value(X::PositiveDefinite) = A_At(vec_to_tril(X.L))

function flatten(::Type{T}, X::PositiveDefinite) where {T<:Real}
    v, unflatten_v = flatten(T, X.L)
    unflatten_PositiveDefinite(v_new::Vector{T}) = PositiveDefinite(unflatten_v(v_new))
    return v, unflatten_PositiveDefinite
end

# Convert a vector to lower-triangular matrix
function vec_to_tril(v::AbstractVector{T}) where {T}
    n_vec = length(v)
    n_tril = Int((sqrt(1 + 8 * n_vec) - 1) / 2) # Infer the size of the matrix from the vector
    L = zeros(T, n_tril, n_tril)
    L[tril!(trues(size(L)))] = v
    return L
end

function ChainRulesCore.rrule(::typeof(vec_to_tril), v::AbstractVector{T}) where {T}
    L = vec_to_tril(v)
    pullback_vec_to_tril(Δ) = NoTangent(), tril_to_vec(unthunk(Δ))
    return L, pullback_vec_to_tril
end

# Convert a lower-triangular matrix to a vector (without the zeros)
# Adapted from https://stackoverflow.com/questions/50651781/extract-lower-triangle-portion-of-a-matrix
function tril_to_vec(X::AbstractMatrix{T}) where {T}
    n, m = size(X)
    n == m || error("Matrix needs to be square")
    return X[tril!(trues(size(X)))]
end
