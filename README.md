# ParameterHandling

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/ParameterHandling.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/ParameterHandling.jl/dev)
[![Build Status](https://travis-ci.com/invenia/ParameterHandling.jl.svg?branch=master)](https://travis-ci.com/invenia/ParameterHandling.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/invenia/ParameterHandling.jl?svg=true)](https://ci.appveyor.com/project/invenia/ParameterHandling-jl)
[![Codecov](https://codecov.io/gh/invenia/ParameterHandling.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/invenia/ParameterHandling.jl)
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

The approach to implementing these types typically revolves around some kind of deferred /
delayed computation. For example, a `Positive` parameter is represented by an
"unconstrained" number, and a "transform" that maps from the entire real line to the
positive half. The `value` of a `Positive` is given by the application of this transform to
the unconstrained number. `flatten`ing a `Positive` yields a length-1 vector containing the
_unconstrained_ number, rather than the value represented by the `Positive` object.
