using RecipesBase
using LaTeXStrings
@recipe function f(r::Divergences.AbstractDivergence; min_u = 0, max_u = 3, lenout = 1000)
    # set a default value for an attribute with `-->`
    xlabel --> L"$u$"
    yguide --> L"$\gamma(u)$"
    #markershape --> :diamond
    # add a series for an error band
    step = (max_u - min_u) / lenout
    u = collect(min_u:step:max_u)
    y = r.(u)
    @series begin
        # force an argument with `:=`
        seriestype := :path
        # ignore series in legend and color cycling
        primary := false
        linecolor := nothing
        #fillcolor := :lightgray
        #fillalpha := 0.5
        #fillrange := r.y .- r.ε
        # ensure no markers are shown for the error band
        markershape := :none
        # return series data
        u, y
    end
    # get the seriescolor passed by the user
    c = get(plotattributes, :seriescolor, :auto)
    # highlight big errors, otherwise use the user-defined color
    #markercolor := ifelse.(r.ε .> ε_max, :red, c)
    # return data
    return u, y
end
