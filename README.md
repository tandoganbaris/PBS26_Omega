# PBS26_Omega

A heuristic for priority-based storage (PBS) escort-routing / load retrieval
in puzzle-based warehouse grids — single and multiple I/O points.

## Files

- `move.jl` — the heuristic: escort assignment and movement.
- `main.jl` — the solve loop: batches loads, calls into `move.jl` each
  iteration, reports makespan and per-load flowtime.
- `structs.jl` — the `item` and `escort` data types.
- `pbsviz.jl` — plotting of the grid state (used by `test.jl`).
- `test.jl` — runs a CSV of instances through the heuristic and writes the
  results back out (see below for what to edit).

## Requirements

Julia, with:

```julia
] add CSV DataFrames DataStructures Plots
```

(`Test`, `Random`, `Statistics`, `Distributed` ship with Julia and need no install.)

## Running `test.jl`

`test.jl` is written against one specific local setup. To run it on your own
machine, change these three hardcoded paths:

| Line | What it is | Change to |
|---|---|---|
| 32 | `CSV.read(raw"C:\codestuff\PBS\FourLoads_escortflow.csv", DataFrame)` | path to your input CSV of instances |
| 59 | `joinpath(raw"C:\codestuff\PBS\plots", string(id_str))` | a folder where per-instance plot subfolders get created |
| 160 | `CSV.write(raw"C:\codestuff\PBS\4loadstestleaveP2.csv", df)` | where the results CSV (with `makespan_heuristic` / `flowtime_heuristic` columns appended) gets written |

The input CSV needs these columns:

- `Lx x Ly` — grid size as a string, e.g. `"10x10"`.
- `IOs`, `Escorts`, `Target Loads` — coordinate lists in the form
  `"{<x1 y1> <x2 y2> ...}"` (0-indexed; `test.jl` converts to Julia's 1-indexing).
- `Retrieval Mode` — `"continue"` or `"leave"` (passed to `main`'s `mode` kwarg).
- `id` — optional. If missing, `test.jl` synthesizes one from
  `Lx x Ly`, `# Escorts`, `#Loads`, and `seed`, so those four columns are
  required instead.

Then:

```
julia test.jl
```

Each row is solved `REPS_PER_ROW` times (10 when running multithreaded via
`julia --threads N test.jl`, else 1) and the best makespan/flowtime per row
is kept. Results are written to the output CSV path above.

## Using your own instance directly

Call `main` without going through a CSV:

```julia
_, makespandict, makespan = main(
    initialstate,   # Lx x Ly Matrix{String}: "" for empty, else an item/escort id
    items,          # Dict{String, item}
    escorts,        # Dict{String, escort}
    IO,             # Tuple{Int,Int} for one I/O point, or Vector{Tuple{Int,Int}} for several
    1,              # run id (used only in plot filenames)
    save_directory; # where plots are written
    n = length(items), no_cores = 1, mode = "continue", mm = "lm"
)
```

`makespandict` maps each load's id to the iteration it was delivered;
`sum(values(makespandict))` is the flowtime. `mm = "lm"` runs the
load-movement variant (one cell per direction per iteration); `mm = "bm"`
runs the batch-movement variant.
