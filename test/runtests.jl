using Bijectors
using Compat: only
using Optim
using ParameterHandling
using Test
using Zygote

using ParameterHandling: value
using ParameterHandling.TestUtils: test_flatten_interface, test_parameter_interface

@testset "ParameterHandling.jl" begin
    include("flatten.jl")
    include("parameters.jl")
end
