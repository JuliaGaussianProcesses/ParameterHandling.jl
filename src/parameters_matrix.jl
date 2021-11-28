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
