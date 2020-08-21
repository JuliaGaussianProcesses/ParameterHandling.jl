using ParameterHandling
using Test

using ParameterHandling.TestUtils: test_flatten_interface

@testset "ParameterHandling.jl" begin
	include("flatten.jl")
    include("parameters.jl")
end
