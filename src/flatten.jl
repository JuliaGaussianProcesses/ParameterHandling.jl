"""
    flatten([eltype=Float64], x)

Returns a "flattened" representation of `x` as a vector of real numbers, and a function
`unflatten` that takes a vector of reals of the same length and returns an object of the
same type as `x`.

`unflatten` is the inverse of `flatten`, so
```julia
julia> x = (randn(5), 5.0, (a=5.0, b=randn(2, 3)));

julia> v, unflatten = flatten(x);

julia> x == unflatten(v)
true
```
"""
function flatten end

flatten(x) = flatten(Float64, x)

function ParameterHandling.flatten(::Type{T}, ::Nothing) where {T<:Real} 
    v = T[]
    unflatten_to_Nothing(::Vector{T}) = nothing
    return v, unflatten_to_Nothing
end

function flatten(::Type{T}, x::Integer) where {T<:Real}
    v = T[]
    unflatten_to_Integer(v::Vector{T}) = x
    return v, unflatten_to_Integer
end

function flatten(::Type{T}, x::R) where {T<:Real,R<:Real}
    v = T[x]
    unflatten_to_Real(v::Vector{T}) = convert(R, only(v))
    return v, unflatten_to_Real
end

function flatten(::Type{T}, x::Vector{R}) where {T<:Real,R<:Real}
    unflatten_to_Vector(v::Vector{T}) = convert(Vector{R}, v)
    return Vector{T}(x), unflatten_to_Vector
end

function _flatten_vector_integer(::Type{T}, x::AbstractVector{<:Integer}) where {T<:Real}
    unflatten_to_Vector_Integer(x_vec) = x
    return T[], unflatten_to_Vector_Integer
end

flatten(::Type{T}, x::Vector{<:Integer}) where {T<:Real} = _flatten_vector_integer(T, x)

function flatten(::Type{T}, x::AbstractVector{<:Integer}) where {T<:Real}
    return _flatten_vector_integer(T, x)
end

function flatten(::Type{T}, x::AbstractVector) where {T<:Real}
    x_vecs_and_backs = map(val -> flatten(T, val), x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = _cumsum(map(length, x_vecs))
        x_Vec = [
            backs[n](x_vec[(sz[n] - length(x_vecs[n]) + 1):sz[n]]) for n in eachindex(x)
        ]
        return oftype(x, x_Vec)
    end
    return reduce(vcat, x_vecs), Vector_from_vec
end

function flatten(::Type{T}, x::AbstractArray) where {T<:Real}
    x_vec, from_vec = flatten(T, vec(x))
    Array_from_vec(x_vec) = oftype(x, reshape(from_vec(x_vec), size(x)))
    return x_vec, Array_from_vec
end

function flatten(::Type{T}, x::SparseMatrixCSC) where {T<:Real}
    x_vec, from_vec = flatten(T, x.nzval)
    Array_from_vec(x_vec) = SparseMatrixCSC(x.m, x.n, x.colptr, x.rowval, from_vec(x_vec))
    return x_vec, Array_from_vec
end

function flatten(::Type{T}, x::Tuple) where {T<:Real}
    x_vecs_and_backs = map(val -> flatten(T, val), x)
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(length, x_vecs)
    sz = _cumsum(lengths)
    function unflatten_to_Tuple(v::Vector{T})
        map(x_backs, lengths, sz) do x_back, l, s
            return x_back(v[(s - l + 1):s])
        end
    end
    return reduce(vcat, x_vecs), unflatten_to_Tuple
end

function flatten(::Type{T}, x::NamedTuple) where {T<:Real}
    x_vec, unflatten = flatten(T, values(x))
    function unflatten_to_NamedTuple(v::Vector{T})
        v_vec_vec = unflatten(v)
        return typeof(x)(v_vec_vec)
    end
    return x_vec, unflatten_to_NamedTuple
end

function flatten(::Type{T}, d::Dict) where {T<:Real}
    d_vec, unflatten = flatten(T, collect(values(d)))
    function unflatten_to_Dict(v::Vector{T})
        v_vec_vec = unflatten(v)
        return Dict(key => v_vec_vec[n] for (n, key) in enumerate(keys(d)))
    end
    return d_vec, unflatten_to_Dict
end

_cumsum(x) = cumsum(x)
if VERSION < v"1.5"
    _cumsum(x::Tuple) = (_cumsum(collect(x))...,)
end

"""
    value_flatten([eltype=Float64], x)

Operates similarly to `flatten`, but the returned `unflatten` function returns an object
like `x`, but with unwrapped values.

Doing
```julia
v, unflatten = value_flatten(x)
```
is the same as doing
```julia
v, _unflatten = flatten(x)
unflatten = ParameterHandling.value ∘ _unflatten
```
"""
function value_flatten(args...)
    v, unflatten = flatten(args...)
    return v, value ∘ unflatten
end
