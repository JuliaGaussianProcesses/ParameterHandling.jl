using Documenter, ParameterHandling

makedocs(;
    modules=[ParameterHandling],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/invenia/ParameterHandling.jl/blob/{commit}{path}#L{line}",
    sitename="ParameterHandling.jl",
    authors="Invenia Technical Computing Corporation",
    assets=[
        "assets/invenia.css",
        "assets/logo.png",
    ],
)

deploydocs(;
    repo="github.com/invenia/ParameterHandling.jl",
)
