using Documenter
using Divergences

DocMeta.setdocmeta!(Divergences, :DocTestSetup, :(using Divergences); recursive = true)

makedocs(
    sitename = "Divergences.jl",
    modules = [Divergences],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://gragusa.github.io/Divergences.jl"
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Divergences" => "divergences.md",
        "Computation" => "computation.md",
        "API Reference" => "api.md"
    ],
    doctest = true,
    checkdocs = :exports,
    warnonly = [:missing_docs]
)

deploydocs(
    repo = "github.com/gragusa/Divergences.jl.git",
    devbranch = "master"
)
