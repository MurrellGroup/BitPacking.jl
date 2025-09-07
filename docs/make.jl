using BitPacking
using Documenter

DocMeta.setdocmeta!(BitPacking, :DocTestSetup, :(using BitPacking); recursive=true)

makedocs(;
    modules=[BitPacking],
    authors="Anton Oresten <antonoresten@proton.me> and contributors",
    sitename="BitPacking.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/BitPacking.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/BitPacking.jl",
    devbranch="main",
)
