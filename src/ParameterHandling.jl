module ParameterHandling

using Compat: only
using ChainRulesCore
using InverseFunctions: inverse
using LogExpFunctions: logit, logistic
using LinearAlgebra
using SparseArrays

export flatten,
    value_flatten, positive, bounded, fixed, deferred, orthogonal, positive_definite

include("flatten.jl")
include("parameters_base.jl")
include("parameters_meta.jl")
include("parameters_scalar.jl")
include("parameters_matrix.jl")
include("parameters_array.jl")

include("test_utils.jl")

end # module
