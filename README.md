# PBS26_Omega

A heuristic for priority-based storage (PBS) escort-routing / load retrieval
in puzzle-based warehouse grids — single and multiple I/O points.

## Files

- `move.jl` — the heuristic: escort assignment and movement.
- `main.jl` — the solve loop: batches loads, calls into `move.jl` each
  iteration, reports makespan and per-load flowtime.
- `structs.jl` — the `item` and `escort` data types.
- `pbsviz.jl` — plotting of the grid state (optional, used by `test.jl`).
- `test.jl` — a minimal runnable example.

## Requirements

Julia, with:

```julia
] add DataStructures Plots
```

(`Test`, `Random`, `Statistics`, `Distributed` ship with Julia and need no install.)

## Running the example

Open `test.jl` and change `SAVE_DIR` at the top to a folder on your machine, then:

```
julia test.jl
```

This builds a small instance (one I/O point, 5 escorts, 3 loads on a 10x10
grid), runs the heuristic, and prints:

- **Makespan** — iterations until every load reaches its I/O point.
- **Flowtime** — sum of each load's own delivery iteration.

A plot of the final grid state is written to `SAVE_DIR`.

## Using your own instance

Call `main` directly:

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

`mm = "lm"` runs the load-movement variant (one cell per direction per
iteration); `mm = "bm"` runs the batch-movement variant.
