# test.jl — minimal runnable example for the PBS escort-routing heuristic.
#
# Edit SAVE_DIR below to a folder on your machine, then run:
#   julia test.jl
#
# Builds one small instance (a 10x10 grid, one I/O point, 5 escorts,
# 3 loads), runs the heuristic, and reports makespan and flowtime.

using Test
include("main.jl")

const SAVE_DIR = raw"C:\path\to\your\plots"   # <-- replace with a real folder; plots are saved here

isdir(SAVE_DIR) || mkpath(SAVE_DIR)
global saveplot = true

@testset "PBS heuristic demo" begin
    Lx, Ly = 10, 10
    IO = (1, 1)   # single I/O point at the grid corner

    escort_coords = [(3, 3), (5, 5), (7, 2), (2, 8), (8, 8)]
    item_coords   = [(4, 4), (6, 6), (9, 9)]

    escorts = Dict{String, escort}()
    for (k, c) in enumerate(escort_coords)
        escorts["E$k"] = escort("E$k", c, String[], String[], 0,
                                 Dict{Int64,Vector{String}}(), Tuple{Int64,Int64}[])
    end

    items = Dict{String, item}()
    for (k, c) in enumerate(item_coords)
        items["I$k"] = item("I$k", c, 0, 0, 1000.0, 1, nothing)
    end

    initialstate = fill("0", Lx, Ly)
    for (key, e) in escorts; x, y = e.coords; initialstate[x, y] = key; end
    for (key, it) in items;  x, y = it.coords; initialstate[x, y] = key; end

    _, makespandict, makespan = main(
        initialstate, items, escorts, IO, 1, SAVE_DIR;
        n = length(items), no_cores = 1, mode = "continue", mm = "lm"
    )

    flowtime = sum(values(makespandict))

    println("Makespan: ", makespan)
    println("Flowtime: ", flowtime)
    println("Per-load delivery iteration: ", makespandict)

    @test makespan < 10_000   # every load reached its I/O point before the iteration cap
end
