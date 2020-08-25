module ParameterHandling

using Bijectors
using Compat: only

export flatten, Positive, Fixed

include("flatten.jl")
include("parameters.jl")

include("test_utils.jl")

end # module
