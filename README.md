# ParameterHandling

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGaussianProcesses.github.io/ParameterHandling.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGaussianProcesses.github.io/ParameterHandling.jl/dev)
[![CI](https://github.com/JuliaGaussianProcesses/ParameterHandling.jl/workflows/CI/badge.svg)](https://github.com/JuliaGaussianProcesses/ParameterHandling.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/JuliaGaussianProcesses/ParameterHandling.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGaussianProcesses/ParameterHandling.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

ParameterHandling.jl is an experiment in handling constrained tunable parameters of models.





# The Parameter Handling Problem

Consider the following common situation: you have a function `build_model` that maps a
collection of parameters `θ` to a `model` of some kind:
```julia
model = build_model(θ)
```
The `model` might, for example, be a function that maps some input `x` to some sort of
prediction `y`:
```julia
y = model(x)
```
where `x` and `y` could essentially be anything that you like.
You might also wish to somehow "learn" or "tune" or "infer" the parameters `θ` by plugging
`build_model` into some other function, lets call it `learn`, that tries out various
different parameter values in some clever way and determines which ones are good -- think
loss minimisation / objective maximisation, (approximate) Bayesian inference, etc.
We'll not worry about exactly what procedure `learn` employs to try out a number of
different parameter values, but suppose that `learn` has the interface:
```julia
learned_θ = learn(build_model, initial_θ)
```

So far so good, but now consider how one actually goes about writing `build_model`.
There are more or less two things that must be written:

1. `θ` must be in a format that `learn` knows how to handle. A popular approach is to
    require that `θ` be a `Vector` of `Real` numbers -- or, rather, some concrete subtype of
    `Real`.
1. The code required to turn `θ` into `model` inside `build_model` mustn't be too onerous to
	write, read, or modify.

While the first point is fairly straightforward, the second point is a bit subtle, so it's
worth dwelling on it a little.

For the sake of concreteness, let's suppose that we adopt the convention that `θ` is a
`Vector{Float64}`. In the case of linear regression, we might assume that `θ` comprises
a length `D` "weight vector" `w`, and a scalar "bias" `b`. So the function to build the
model might be something like

```julia
function build_model(θ::Vector{Float64})
    return x -> dot(θ[1:end-1], x) + θ[end]
end
```

The easiest way to see that this is a less than ideal solution is to consider what this
function would look like if `θ` was, say, a `NamedTuple` with fields `w` and `b`:
```julia
function build_model(θ::NamedTuple)
    return x -> dot(θ.w, x) + θ.b
end
```
This version of the function is much easier to read -- moreover if you want to inspect the
values of `w` and `b` at some other point in time, you don't need to know precisely how to
chop up the vector.

Moreover it seems probable that the latter approach is less
bug-prone -- suppose for some reason one refactored the code so that the first element of
`θ` became `b` and the last `D` elements `w`; any code that depended upon the original
ordering will now be incorrect and likely fail silently. The `NamedTuple` approach simply
doesn't have this issue.

Granted, in this simple case it's not too much of a problem, but it's easy to find
situations in which things become considerably more difficult. For example, suppose that we
instead had pretty much any kind of neural network, Gaussian process, ODE, or really just
any model with more than a couple of distinct parameters. From the perspective of
writing complicated models, implementing things in terms of a single vector of
parameters that is _manually_ chopped up is an _extremely_ bad design choice. It simply
doesn't scale.

However, a single vector of e.g. `Float64`s _is_ extremely convenient when writing general
purpose optimisers / approximate inference routines --
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) and
[AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) being two obvious examples.





# The ParameterHandling.jl Approach

`ParameterHandling.jl` aims to give you the best of both worlds by providing the tools
required to automate the transformation between a "structured" representation (e.g. nested
`NamedTuple` / `Dict` etc) and a "flattened" (e.g. `Vector{Float64}`) of your model
parameters.

The function `flatten` eats a structured representation of some parameters, returning the
flattened representation _and_ a function that converts the flattened thing back into its
structured representation.

`flatten` is implemented recursively, with a _very_ small number of base-implementations
that don't themselves call `flatten`.

You should expect to occassionally have to extend `flatten` to handle your own types and, if
you wind up doing this for a function in `Base` that this package doesn't yet cover, a PR
including that implementation will be very welcome.

See `test/parameters.jl` for a couple of examples that utilise `flatten` to do something
similar to the task described above.





# Dealing with Constrained Parameters

It is very common to need to handle constraints on parameters e.g. it may be necessary for a
particular scalar to always be positive. While `flatten` is great for changing between
representations of your parameters, it doesn't really have anything to say about this
constraint problem.

For this we introduce a collection of new `AbstractParameter` types (whether we really need
them to have some mutual supertype is unclear at present) that play nicely with `flatten`
and allow one to specify that e.g. a particular scalar must remain positive, or should be
fixed across iterations. See `src/parameters.jl` and `test/parameters.jl` for more examples.

The approach to implementing these types typically revolves around some kind of `Deferred` /
delayed computation. For example, a `Positive` parameter is represented by an
"unconstrained" number, and a "transform" that maps from the entire real line to the
positive half. The `value` of a `Positive` is given by the application of this transform to
the unconstrained number. `flatten`ing a `Positive` yields a length-1 vector containing the
_unconstrained_ number, rather than the value represented by the `Positive` object. For
example

```julia
julia> using ParameterHandling

julia> x_constrained = 1.0 # Specify constrained value.
1.0

julia> x = positive(x_constrained) # Construct a number that should remain positive.
ParameterHandling.Positive{Float64, typeof(exp), Float64}(-1.490116130486996e-8, exp, 1.4901161193847656e-8)

julia> ParameterHandling.value(x) # Get the constrained value by applying the transform.
1.0

julia> v, unflatten = flatten(x); # Supports the `flatten` interface.

julia> v
1-element Vector{Float64}:
 -1.490116130486996e-8

julia> new_v = randn(1) # Pick a random new value.
1-element Vector{Float64}:
 2.3482666974328716

julia> ParameterHandling.value(unflatten(new_v)) # Obtain constrained value.
10.467410816707215
```

We also provide the utility function `value_flatten` which returns an unflattening function
equivalent to `value(unflatten(v))`. The above could then be implemented as
```julia
julia> v, unflatten = value_flatten(x);

julia> unflatten(v)
1.0
```

It is straightforward to implement your own parameters that interoperate with those already
written by implementing `value` and `flatten` for them. You might want to do this if this
package doesn't currently support the functionality that you need.




# A Worked Example



We use a model involving a Gaussian process (GP) -- you don't need to know anything about
Gaussian processes other than
1. they are a class of probabilistic model which can be used for regression (amongst other things).
2. they have some tunable parameters that are usually chosen by optimising a scalar objective function using an iterative
optimisation algorithm -- typically a variant of gradient descent.
Ths is representative of a large number of models in ML / statistics / optimisation.

This example can be copy+pasted into a REPL session.

```julia
# Install some packages.
using Pkg
Pkg.add("ParameterHandling")
Pkg.add("Optim")
Pkg.add("Zygote")
Pkg.add("AbstractGPs")

using ParameterHandling # load up this package.
using Optim # generic optimisation
using Zygote # algorithmic differentiation
using AbstractGPs # package containing the models we'll be working with

# Declare a NamedTuple containing an initial guess at parameters.
raw_initial_params = (
    k1 = (
        var=positive(0.9),
        precision=positive(1.0),
    ),
    k2 = (
        var=positive(0.1),
        precision=positive(0.3),
    ),
    noise_var=positive(0.2),
)

# Using ParameterHandling.value_flatten, we can obtain both a Vector{Float64} representation of
# these parameters, and a mapping from that vector back to the original (unconstrained) parameter values.
flat_initial_params, unflatten = ParameterHandling.value_flatten(raw_initial_params)

# ParameterHandling.value strips out all of the Positive types in initial_params,
# returning a plain named tuple of named tuples and Float64s.
julia> initial_params = ParameterHandling.value(raw_initial_params)
(k1 = (var = 0.9, precision = 1.0), k2 = (var = 0.10000000000000002, precision = 0.30000000000000004), noise_var = 0.19999999999999998)

# GP-specific functionality. Don't worry about the details, just
# note the use of the structured representation of the parameters.
function build_gp(params::NamedTuple)
    k1 = params.k1.var * Matern52Kernel() ∘ ScaleTransform(params.k1.precision)
    k2 = params.k2.var * SEKernel() ∘ ScaleTransform(params.k2.precision)
    return GP(k1 + k2)
end

# Generate some synthetic training data.
# Again, don't worry too much about the specifics here.
const x = range(-5.0, 5.0; length=100)
const y = rand(build_gp(initial_params)(x, initial_params.noise_var))

# Specify an objective function in terms of x and y.
function objective(params::NamedTuple)
    f = build_gp(params)
    return -logpdf(f(x, params.noise_var), y)
end

# Use Optim.jl to minimise the objective function w.r.t. the params.
# The important thing here is to note that we're passing in the flat vector of parameters to
# Optim, which is something that Optim knows how to work with, and using `unflatten` to convert
# from this representation to the structured one that our objective function knows about
# using `unflatten` -- we've used ParameterHandling to build a bridge between Optim and an
# entirely unrelated package.
training_results = Optim.optimize(
    objective ∘ unflatten,
    θ -> only(Zygote.gradient(objective ∘ unflatten, θ)),
    flat_initial_params,
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(show_trace = true);
    inplace=false,
)

# Extracting the final values of the parameters.
final_params = unflatten(training_results.minimizer)
f_trained = build_gp(final_params)
```

Usually you would go on to make some predictions on test data using `f_trained`, or
something like that.
From the perspective of ParameterHandling.jl, we've seen the interesting stuff though.
In particular, we've seen an example of how ParameterHandling.jl can be used to bridge the
gap between the "flat" representation of parameters that `Optim` likes to work with, and the
"structured" representation that it's convenient to write optimisation algorithms with.

# Gotchas and Performance Tips

1. `Integer`s typically don't take part in the kind of optimisation procedures that this package is designed to handle. Consequently, `flatten(::Integer)` produces an empty vector.
2. `deferred` has some type-stability issues when used in conjunction with abstract types. For example, `flatten(deferred(Normal, 5.0, 4.0))` won't infer properly. A simple work around is to write a function `normal(args...) = Normal(args...)` and work with `deferred(normal, 5.0, 4.0)` instead.
3. Let `x` be an `Array{<:Real}`. If you wish to constrain each of its values to be positive, prefer `positive(x)` over `map(positive, x)` or `positive.(x)`. `positive(x)` has been implemented the associated `unflatten` function has good performance, particularly when interacting with `Zygote` (when `map(positive, x)` is extremely slow). The same thing applies to `bounded` values. Prefer `bounded(x, lb, ub)` to e.g. `bounded.(x, lb, ub)`.
