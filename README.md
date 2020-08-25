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
There are a considerations that must be made:

1. `θ` must be in a format that `learn` knows how to handle. A popular approach is to
    require that `θ` be a `Vector` of `Real` numbers -- or, rather, some concrete subtype of
    `Real`. 
1. The code required to turn `θ` into `model` inside `build_model` mustn't be too onerous.

While the first point pretty straightforward, the second point is a bit subtle, so it's
worth working through a couple of examples to understand what it's getting at.

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

Moreover it seems quite probably that the latter approach is less
bug-prone -- suppose for some reason one refactored the code so that the first element of
`θ` became `b` and the last `D` elements `w`; any code that depended upon the original
ordering will now be incorrect and likely fail silently. The `NamedTuple` approach simply
doesn't have this issue.

Granted, in this simple case it's not too much of a problem, but it's easy to find
situations in which things become considerably more difficult. For example, suppose that we
instead had pretty much any kind of neural network, Gaussian process, ODE, or really just
any model with more than a couple of distinct parameters. From the perspective of
implementing writing complicated models, implementing things in terms of a single vector of
parameters that is _manually_ chopped up is an _extremely_ bad design choice. It simply
doesn't scale.

However, a single vector of e.g. `Float64`s _is_ extremely convenient when writing general
purpose optimisers / approximate inference routines --
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) and
[AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) being two obvious examples.





# The ParameterHandling.jl Approach
