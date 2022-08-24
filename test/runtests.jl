using Compat: only
using ChainRulesTestUtils
using Distributions
using LinearAlgebra
using Optim
using ParameterHandling
using PDMats
using Test
using Zygote
using SparseArrays

using ParameterHandling: value
using ParameterHandling.TestUtils: test_flatten_interface, test_parameter_interface

const tuple_infers = VERSION < v"1.5" ? false : true

@testset "ParameterHandling.jl" begin
    include("flatten.jl")
    include("parameters.jl")
    include("parameters_meta.jl")
    include("parameters_scalar.jl")
    include("parameters_matrix.jl")
    include("parameters_array.jl")
end
