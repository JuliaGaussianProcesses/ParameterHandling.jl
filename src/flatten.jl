"""
    flatten(x)

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

function flatten(x::Real)
    v = [x]
    unflatten_to_Real(v::Vector{<:Real}) = only(v)
    return v, unflatten_to_Real
end

flatten(x::Vector{<:Real}) = (x, identity)

function flatten(x::AbstractVector)
    x_vecs_and_backs = map(flatten, x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = cumsum(map(length, x_vecs))
        x_Vec = [backs[n](x_vec[sz[n] - length(x_vecs[n]) + 1:sz[n]]) for n in eachindex(x)]
        return oftype(x, x_Vec)
    end
    return vcat(x_vecs...), Vector_from_vec
end

function flatten(x::AbstractArray)

    x_vec, from_vec = flatten(vec(x))

    function Array_from_vec(x_vec)
        return oftype(x, reshape(from_vec(x_vec), size(x)))
    end

    return x_vec, Array_from_vec
end

function flatten(x::Tuple)
    x_vecs, unflattens = zip(map(flatten, x)...)
    sz = cumsum(collect(map(length, x_vecs)))
    function unflatten_to_Tuple(v::Vector{<:Real})
        return ntuple(length(x)) do n
            return unflattens[n](v[sz[n] - length(x_vecs[n]) + 1:sz[n]])
        end
    end
    return reduce(vcat, x_vecs), unflatten_to_Tuple
end

function flatten(x::NamedTuple)
    x_vec, unflatten = flatten(values(x))
    function unflatten_to_NamedTuple(v::Vector{<:Real})
        v_vec_vec = unflatten(v)
        return typeof(x)(v_vec_vec)
    end
    return x_vec, unflatten_to_NamedTuple
end

function flatten(d::Dict)
    d_vec, unflatten = flatten(collect(values(d)))
    function unflatten_to_Dict(v::Vector{<:Real})
        v_vec_vec = unflatten(v)
        return Dict(key => v_vec_vec[n] for (n, key) in enumerate(keys(d)))
    end
    return d_vec, unflatten_to_Dict
end
