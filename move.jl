using Statistics
using DataStructures
using Random

# Tunable GRASP alpha for the item re-sort inside item_escort_assigment_r! (single-IO
# randomized assignment). Exposed as a mutable Ref so callers (e.g. parameteropt.jl) can
# sweep it without threading a new kwarg through PBSengine! -> item_escort_assigment_r!.
const GRASP_ITEM_ALPHA = Ref(0.9)

# Tunable on/off for the zigzag interleave in item_escort_assigment! (single-IO) and
# item_escort_IO_assigment! (multi-IO) deterministic assignment. Exposed as a mutable
# Ref so callers (e.g. test scripts) can A/B it without threading a new kwarg through
# PBSengine! -> main.jl. Controls both IO cases with one shared switch.
const ZIGZAG_INTERLEAVE = Ref(true)

# "Load movement" (LM) mode: when true, move_escort! clamps every move to a
# single cell in the chosen direction instead of shoving the full distance in
# one call. Set by main()'s `mm` kwarg (mm=="lm") for the duration of that
# run; every caller of move_escort! (assignment movers, direct/urgent serve,
# freeroam, cooperative freeroam) is unaffected code-wise — each just takes
# many more, smaller steps to reach the same target over subsequent
# iterations. Left false ("BM", batch movement) preserves existing behavior.
const UNIT_STEP = Ref(false)

# Total count of successful (1-cell, when UNIT_STEP) move_escort! calls in
# the current main() run — this is the "number of moves" metric comparable
# to the He/RL/paper datasets, which count individual escort step-events.
# Reset to 0 at the start of every main() call; read it right after main()
# returns (single-threaded callers only — no_cores>1 parallel replicates
# would race on this counter, so it's only meaningful for no_cores==1 runs).
const TOTAL_MOVES = Ref(0)
const LM_PATHCHECK = Ref(true)   # run mover-phase path_to_io_exists_if in LM too (fixes corner-IO deadlocks; small makespan cost). Toggle for A/B.
# ESCORT_RELOCS: under lm, one per single-cell escort relocation, EXCEPT a
# relocation onto a cell already held by another escort (two empties swapping,
# which displaces nothing). Under bm it just mirrors TOTAL_MOVES. Reset in
# main() alongside TOTAL_MOVES.
const ESCORT_RELOCS = Ref(0)

# LM mode only: once an escort starts actively serving an item, keep it
# committed to that item across iterations instead of letting
# find_nearest_escort re-pick "nearest" from scratch every call. Under 1-cell
# stepping, two similarly-placed escorts' relative distances to an item can
# flip after almost every move, causing find_nearest_escort to swap which one
# is "nearest" every iteration — both escorts then spend moves re-approaching
# instead of one of them ever finishing the delivery. Keyed by item id,
# cleared at the start of every main() run.
const COMMITTED_ESCORT = Dict{String,String}()

# Diagnostic logging for the mover-phase "candid" detour investigation:
# whenever moveescorts_flow_multi_io!'s mover phase finds a candidate item
# (candid) in the way of the escort's own assigned item and does NOT just
# head straight for the assigned item's coords, log the event. Gated by
# LOG_MOVER_CANDID[] so it's zero-cost when off.
const LOG_MOVER_CANDID = Ref(false)
const MOVER_CANDID_LOG = NamedTuple[]

# A/B knob for the candid-detour fix (see MOVER_CANDID_LOG investigation):
# find_nearest_item_toitem's direction==2 branch never checks the "other
# item" is actually in the same column as the escort's own assigned item
# (direction==1 has the symmetric bug on rows) -- it just grabs whichever
# item has the next-largest y, however far away in x, and the mover phase
# then aims partway toward THAT item's row instead of continuing straight to
# its own item, effectively wandering off toward an unrelated item.
#   0 = off (original behavior)
#   1 = ignore candid entirely in the mover phase -- always head straight for
#       the assigned item's own coords
#   2 = fix at the source: find_nearest_item_toitem only returns a candidate
#       that's in the exact same column (dir 2) / row (dir 1)
#   3 = same as 2 but with a +-1 cell alignment tolerance instead of exact
const CANDID_FIX_MODE = Ref(0)
# LM mode, multi-IO only: once an item starts being served toward a particular IO,
# pin that IO as the item's target on subsequent iterations instead of recomputing
# the nearest IO. Without this, an item sitting between two IOs flip-flops its
# target every step (move it toward IO-A, now IO-B is nearer, move it back), and a
# lone escort never finishes either delivery. Keyed by item id, cleared at the
# start of every main() run and pruned when the item is delivered. BM path is
# unaffected (only consulted when UNIT_STEP[]).
const COMMITTED_IO = Dict{String,Tuple{Int,Int}}()
const BASE_ESCORT_REPOSITION = Ref(false)
const DEBUG_MOVE_TRACE = Ref(false)
# Temporary: when non-empty, moveescorts_flow_multi_io! prints which phase
# (MOVER / DIRECTSERVE / URGSERVE / FREEROAM) moves this escort id and its
# from/to coords, each iteration. Used to trace the E2 sidestep.
const TRACE_ESC = Ref("")
# Contributor tracking for the prune-and-rerun experiment: when TRACK_CONTRIB[]
# is true, move_escort! records (itemid -> escortid) whenever a move directly
# displaces that item (the escort's step swaps places with the item, i.e. the
# item is the cell the escort is stepping into). CONTRIB_ESCORTS[itemid] is
# the set of escorts that ever directly moved that item forward. Escorts that
# only ever freeroam/reposition without ever touching an item are absent from
# every set. Reset before each pass-1 run.
const TRACK_CONTRIB = Ref(false)
const CONTRIB_ESCORTS = Dict{String,Set{String}}()
# Also record, per escort, every iteration at which it directly moved some
# item (same trigger as CONTRIB_ESCORTS). Used to build activation windows
# for the freeze-outside-window postprocess pass below.
const ESCORT_CONTRIB_ITERS = Dict{String,Vector{Int}}()
# Per escort, chronological (iteration, itemid) log of every item it directly
# touched -- lets a postprocess pass lock each contributing escort to
# whichever item it touched FIRST ("did escort 1 move item 1, then it
# belongs to it and shouldn't move towards somewhere else").
const ESCORT_CONTRIB_EVENTS = Dict{String,Vector{Tuple{Int,String}}}()
const CURRENT_ITER = Ref(0)
function mark_contrib!(itemid, escid)
    TRACK_CONTRIB[] || return
    push!(get!(CONTRIB_ESCORTS, itemid, Set{String}()), escid)
    push!(get!(ESCORT_CONTRIB_ITERS, escid, Int[]), CURRENT_ITER[])
    push!(get!(ESCORT_CONTRIB_EVENTS, escid, Tuple{Int,String}[]), (CURRENT_ITER[], itemid))
end

# Postprocess pass 2, windowed variant: instead of omitting non-contributing
# escorts entirely (which changes corridor geometry), keep every escort
# physically in the grid but freeze each one outside the iteration window(s)
# around when it actually contributed. An escort with no contribution ever
# is frozen for the whole run (a pure static obstacle). Windows are built by
# the caller into ESCORT_ACTIVE_WINDOWS from ESCORT_CONTRIB_ITERS ± padding.
const FREEZE_ENABLED = Ref(false)
const ESCORT_ACTIVE_WINDOWS = Dict{String,Vector{Tuple{Int,Int}}}()
function escort_active(escid, iter)
    FREEZE_ENABLED[] || return true
    ws = get(ESCORT_ACTIVE_WINDOWS, escid, Tuple{Int,Int}[])
    any(w -> w[1] <= iter <= w[2], ws)
end

# Hard item lock: escort -> the SET of items it's permanently allowed to
# serve (e.g. every item it was ever observed helping in a discovery pass, or
# just its nearest item decided upfront). When ITEM_LOCK_ENABLED[], the
# mover/directserve/urgserve phases (multi-IO) refuse to assign or serve any
# item outside that set — "it belongs to [the items it serves], excluding
# others it doesn't serve." An escort with no entry at all is locked out of
# everything (matches the reclaim logic below).
const ITEM_LOCK_ENABLED = Ref(false)
const ESCORT_ITEM_LOCK = Dict{String,Set{String}}()
item_lock_ok(escid, itemid) = !ITEM_LOCK_ENABLED[] || itemid in get(ESCORT_ITEM_LOCK, escid, Set{String}())

# The (lock-agnostic) assignment phase can still put an escort on the mover
# roster for an item that isn't its lock target. moveescorts_flow_multi_io!'s
# mover loop already refuses to act on it (item_lock_ok guard, `continue`) --
# but being on that roster at all excludes the escort from `nonmovers`, so
# without this check a lock-mismatched escort would be shut out of
# directserve/urgserve/freeroam too and simply do nothing, forever (this was
# the root cause of the 0/N convergence collapse under locking). An escort
# only counts as a "real" mover under locking if at least one of its
# assignments actually targets its locked item.
function mover_matches_lock(escortid, global_escort_items)
    ITEM_LOCK_ENABLED[] || return true
    haskey(global_escort_items, escortid) || return false
    any(global_escort_items[escortid]) do assignment
        _, itemsx, itemsy = assignment
        (!isempty(itemsx) && item_lock_ok(escortid, itemsx[1])) ||
        (!isempty(itemsy) && item_lock_ok(escortid, itemsy[1]))
    end
end
# LM multi-IO: max IO-zone index gap between an escort and an urgent load's
# target IO for that load to be eligible for urgent-serve by that escort.
# 1 = own IO + immediate neighbours only. A/B knob.
const LM_URG_IO_RADIUS = Ref(3)
# LM multi-IO park gate: how many IO-zones away from a demand zone an idle
# escort's own zone may be before it still counts as "movable". 1 = only the
# immediately adjacent zone; was hardcoded at 1, now tunable for A/B.
const PARK_GATE_RADIUS = Ref(1)
# Order items in item_escort_IO_assigment! by how close their nearest available
# escort is (items with an adjacent escort first, then by increasing distance to
# that nearest escort) instead of by distance to IO. Lets an item with an escort
# right next to it claim that escort before a far-away item can greedily grab it.
# ON by default but only takes effect in LM mode (UNIT_STEP[]); BM keeps the
# distance-to-IO order. Flip to false to A/B it off.
const ESCORT_PROXIMITY_ORDER = Ref(true)
# Diagnostic: space-separated item/escort ids to trace in item_escort_IO_assigment!
const DBG_IOASSIGN = Ref("")
# Diagnostic: dump idle-flex redirect + freeroam detail on this iteration only.
const DBG_ITER = Ref(0)
# LM multi-IO Pass-1 IO ordering: 0 = IO vector order (x-descending, original);
# 1/2 = most/fewest escorts in zone first; 3/4 = most/fewest items in zone
# first. Default 2 -- escort-starved IOs get first pick of the shared pool.
const IO_ORDER_MODE = Ref(2)

# Idle-flex: an item is "idling" when its position has not changed for several
# iterations -- whether because it never got an escort, or because the escort
# it did get is stuck. Once idling, progressively relax the exact row/column
# alignment an escort must have to be a candidate for it, so a nearby-but-
# misaligned escort can be recruited and walked into place, and (past a higher
# threshold) redirect the nearest idle escorts to freeroam toward that item.
# ITEM_IDLE[itemid] = consecutive stagnant iterations; ITEM_LASTPOS tracks the
# previous position. Both reset in main().
const IDLE_FLEX = Ref(false)   # on ONLY for the nearest-1-escort multi-IO/LM runner, which sets IDLE_FLEX[]=true itself; default-on regressed This-paper/k>=2 (freeroam redirect congests when escorts are plentiful)
const IDLE_FLEX_START = Ref(12)      # stagnant iterations before any relaxation
const IDLE_FLEX_REDIRECT = Ref(25)   # stagnant iterations before pulling escorts over by freeroam
const ITEM_IDLE = Dict{String,Int}()
const ITEM_LASTPOS = Dict{String,Tuple{Int,Int}}()
idle_tol(key) = IDLE_FLEX[] ? clamp((get(ITEM_IDLE, key, 0) - IDLE_FLEX_START[]) ÷ 6 + 1, 0, 6) : 0
_dbg_io_watch(x) = DBG_IOASSIGN[] != "" && x in split(DBG_IOASSIGN[])

# LM mode, single-item runs only: once the lone item sits on its assigned IO the
# run is done — any further escort/load step only inflates TOTAL_MOVES (the
# objective) without serving anything. Set by main() (mm=="lm" && exactly one
# item); move_escort! then refuses every subsequent move.
const SINGLE_ITEM_RUN = Ref(false)

# LM mode: force freeroam! to abandon its directional left/up/right ladder and instead
# step to whichever neighbour cell most reduces the A* cost-to-IO map (asternmat), so an
# idle escort can take detours that leave the IO's own row/column (e.g. down-and-under an
# item wedged against the IO). This behaviour is applied AUTOMATICALLY in LM mode whenever
# there is a single escort serving multiple loads (see freeroam!) — the regime where the
# ladder deadlocks (Mirzaei-type) and where the pile-up downside of greedy descent can't
# occur because there is only one escort. This Ref is just a manual override to force it on
# in other regimes for experimentation; default off = automatic gating only.
const FREEROAM_GRADIENT = Ref(false)

"""
assigns escorts to items based on the initial sorting (in the function) and the positions in the matrix
"""
function PBSengine!(iteration, incumbentstate, batch, escorts, IO; obj="makespan", no_cores=1, deterministic_anchor=false)
    if no_cores == 1
        if obj == "makespan"
            if isa(IO, Tuple)
                moved_any = false
                moverescortids, blockmat = item_escort_assigment!(incumbentstate, batch, escorts, iteration, IO)
                moved_any = moveescorts!(iteration, incumbentstate, batch, escorts, moverescortids, blockmat, IO)
                return moved_any
            elseif isa(IO, Vector{Tuple{Int,Int}})
                (global_blockmat, global_escort_items, item_to_ios) = item_escort_IO_assigment!(incumbentstate, batch, escorts, iteration, IO)
                moved_any = moveescorts_flow_multi_io!(iteration, incumbentstate, batch, escorts,
                            global_blockmat, global_escort_items, IO, item_to_ios)
                return moved_any
            else
                throw(ArgumentError("IO in wrong format: should be a Tuple{x=int,y=int} or an Vector{Tuple{Int,Int}}"))
            end
        elseif obj == "flowtime"
            if isa(IO, Tuple)
                moved_any = false
                moverescortids, blockmat = item_escort_assigment!(incumbentstate, batch, escorts, iteration, IO)
                moved_any = moveescorts_flow!(iteration, incumbentstate, batch, escorts, moverescortids, blockmat, IO)
                return moved_any
            elseif isa(IO, Vector{Tuple{Int,Int}})
                (global_blockmat, global_escort_items, item_to_ios) = item_escort_IO_assigment!(incumbentstate, batch, escorts, iteration, IO)
                moved_any = moveescorts_flow_multi_io!(iteration, incumbentstate, batch, escorts,
                            global_blockmat, global_escort_items, IO, item_to_ios)
                return moved_any
            else
                throw(ArgumentError("IO in wrong format: should be a Tuple{x=int,y=int} or an Vector{Tuple{Int,Int}}"))
            end
        end

    else
        # ── Parallel GRASP: run no_cores independent randomised copies, keep best ──
        # Supports both single-IO (Tuple) and multi-IO (Vector{Tuple{Int,Int}}).
        if !isa(IO, Tuple) && !isa(IO, Vector{Tuple{Int,Int}})
            throw(ArgumentError("IO in wrong format: should be a Tuple{x=int,y=int} or an Vector{Tuple{Int,Int}}"))
        end
        multi_io = isa(IO, Vector{Tuple{Int,Int}})

        # Quality proxy: lower is better.
        # Component 1 — total Manhattan distance of each item to its nearest IO (minimise).
        # Component 2 — coverage discount: for each (item, direction) pair where an escort
        #   sits on the direct path from the item to its nearest IO, the total distance is
        #   discounted proportionally. Each item contributes at most 1 per direction (X and
        #   Y independently), so the maximum coverage is 2 × n_items.
        # COVERAGE_WEIGHT caps how much of the distance coverage can discount (< 1.0), so
        # distance always stays the tiebreaker between two states with equal coverage —
        # even at full coverage, a state is never fully indifferent to its own distance.
        COVERAGE_WEIGHT = 0.9
        function step_quality(local_batch, local_escorts)
            escort_coords = Set(e.coords for e in values(local_escorts))
            total_dist = 0.0
            coverage   = 0          # counts (item, direction) pairs that are covered

            for k in keys(local_batch)
                ix, iy = local_batch[k].coords
                iox, ioy = multi_io ? argmin(io -> abs(io[1]-ix)+abs(io[2]-iy), IO) : IO
                total_dist += abs(ix - iox) + abs(iy - ioy)

                # Y-coverage: escort in the same column (x=ix) strictly between item and IO on y
                y_lo, y_hi = minmax(iy, ioy)
                if any((ix, ey) in escort_coords for ey in y_lo+1:y_hi-1)
                    coverage += 1   # item covered in Y direction (at most once)
                end

                # X-coverage: escort in the same row (y=iy) strictly between item and IO on x
                x_lo, x_hi = minmax(ix, iox)
                if any((ex, iy) in escort_coords for ex in x_lo+1:x_hi-1)
                    coverage += 1   # item covered in X direction (at most once)
                end
            end

            n = length(local_batch)
            max_coverage = 2 * n   # each item contributes at most 1 per axis (X and Y)
            coverage_frac = max_coverage > 0 ? coverage / max_coverage : 0.0
            # (1 - COVERAGE_WEIGHT*coverage_frac) ∈ [1-COVERAGE_WEIGHT, 1], always > 0, so
            # this never flips sign AND distance still breaks ties at equal coverage.
            return total_dist * (1 - COVERAGE_WEIGHT * coverage_frac)
        end

        # Deep-copy sequentially BEFORE spawning threads to avoid concurrent deepcopy aliasing
        thread_copies = [(deepcopy(incumbentstate), deepcopy(batch), deepcopy(escorts)) for _ in 1:no_cores]
        results = Vector{Any}(undef, no_cores)
        # When deterministic_anchor is set, slots 1 and 2 are reserved as deterministic
        # anchors (zigzag on, then zigzag off) instead of GRASP-randomized — but only on
        # the caller's first iteration (e.g. the first PBSengine! call of the first rep),
        # so the anchors are sampled exactly once rather than on every time step.
        anchor_this_call = deterministic_anchor && iteration == 1
        guard_io = reach_guard_io()   # task-local, so hand it to the spawned tasks explicitly
        Threads.@threads for i in 1:no_cores
            set_reach_guard!(guard_io)
            local_state, local_batch, local_escorts = thread_copies[i]
            try
                if anchor_this_call && (i == 1 || i == 2)
                    use_zigzag = (i == 1)
                    if multi_io
                        global_blockmat, global_escort_items, item_to_ios = item_escort_IO_assigment!(local_state, local_batch, local_escorts, iteration, IO; use_zigzag=use_zigzag)
                        moved = moveescorts_flow_multi_io!(iteration, local_state, local_batch, local_escorts,
                                    global_blockmat, global_escort_items, IO, item_to_ios)
                    else
                        moverescortids, blockmat = item_escort_assigment!(local_state, local_batch, local_escorts, iteration, IO; use_zigzag=use_zigzag)
                        moved = moveescorts_flow!(iteration, local_state, local_batch, local_escorts, moverescortids, blockmat, IO)
                    end
                else
                    if multi_io
                        global_blockmat, global_escort_items, item_to_ios = item_escort_IO_assigment_r!(local_state, local_batch, local_escorts, iteration, IO)
                        moved = moveescorts_flow_multi_io_r!(iteration, local_state, local_batch, local_escorts,
                                    global_blockmat, global_escort_items, IO, item_to_ios)
                    else
                        moverescortids, blockmat = item_escort_assigment_r!(local_state, local_batch, local_escorts, iteration, IO)
                        moved = moveescorts_flow_r!(iteration, local_state, local_batch, local_escorts, moverescortids, blockmat, IO)
                    end
                end

                quality = step_quality(local_batch, local_escorts)
                results[i] = (moved, quality, local_state, local_batch, local_escorts)
            catch e
                println("Warning: GRASP replicate $i failed at iteration $iteration ($e) — treating as no-move")
                results[i] = (false, Inf, local_state, local_batch, local_escorts)
            end
        end

        # Pick the run where at least one escort moved AND total item distance is smallest.
        # Fall back to lowest quality among non-moving runs if nothing moved.
        moved_runs = filter(i -> results[i][1], 1:no_cores)
        best_idx = if !isempty(moved_runs)
            argmin(i -> results[i][2], moved_runs)
        else
            argmin(i -> results[i][2], 1:no_cores)
        end

        _, _, best_state, best_batch, best_escorts = results[best_idx]

        # Apply the best run's result back to the caller's mutable arguments
        incumbentstate .= best_state
        empty!(batch);   merge!(batch,   best_batch)
        empty!(escorts); merge!(escorts, best_escorts)

        return results[best_idx][1]  # moved_any of the winning run
    end
end

function item_IO_assignment(item, IO)
    if isa(IO, Tuple)
        item.assigned_io = IO
    elseif isa(IO, Vector{Tuple{Int,Int}})
        item.assigned_io = IO
    else
        throw(ArgumentError("IO in wrong format: should be a Tuple{x=int,y=int} or an Vector{Tuple{Int,Int}}"))
    end
end


function item_escort_IO_assigment!(matrix, items, escorts, iteration, IO; use_zigzag=ZIGZAG_INTERLEAVE[])
    # Handle multi-IO case: create separate assignments for each IO
    if isa(IO, Vector{Tuple{Int,Int}})
        io_assignments = Dict()  # Will store: IO => (escortstomovefirst, blockmat, itemescortdict)
        
        resetescorts!(escorts, iteration)  # Reset escorts once for all IOs
        
        # Pre-process: determine which IOs are relevant for each item
        item_to_ios = Dict{String, Vector{Tuple}}()  # item_id => [primary_io, secondary_io (if any)]
        
        for item_id in keys(items)
            item_x, item_y = items[item_id].coords
            
            # Calculate distances to all IOs
            io_distances = [(io, euclidean_distance((item_x, item_y), io)) for io in IO]
            sort!(io_distances, by = x -> x[2])  # Sort by distance
            
            # Keep at most 2 closest IOs, with x-coordinate filtering
            relevant_ios = Tuple{Int,Int}[]
            for (i, (io, dist)) in enumerate(io_distances[1:min(2, length(io_distances))])
                io_x, io_y = io
                should_consider = true
                
                # X-coordinate alignment check
                if i == 1
                    # Always consider the closest IO
                    push!(relevant_ios, io)
                else
                    # For secondary IO, check x-coordinate alignment
                    primary_io_x, _ = io_distances[1][1]
                    
                    # Don't consider secondary IO if item is not between/beyond the IOs
                    if (item_x < min(primary_io_x, io_x)) || (item_x > max(primary_io_x, io_x))
                        # Item is on one side, only one IO is relevant
                        if item_x < min(primary_io_x, io_x)
                            # Item is to the left, keep the leftmost IO (already primary)
                            should_consider = false
                        elseif item_x > max(primary_io_x, io_x)
                            # Item is to the right, replace primary with rightmost if primary is not rightmost
                            if primary_io_x > io_x
                                should_consider = false
                            else
                                # Primary is leftmost, secondary is rightmost - keep both logic
                                push!(relevant_ios, io)
                                should_consider = false
                            end
                        end
                    else
                        # Item is between or near both IOs
                        x_gap = abs(primary_io_x - io_x)
                        x_to_primary = abs(item_x - primary_io_x)
                        x_to_secondary = abs(item_x - io_x)
                        
                        # Check if we already have a secondary IO; if current candidate is closer, replace it
                        if length(relevant_ios) > 1
                            existing_io = relevant_ios[2]
                            existing_io_x, _ = existing_io
                            
                            # If current candidate is closer to item than existing secondary, replace it
                            if x_to_secondary < abs(item_x - existing_io_x)
                                # Replace the secondary IO with the closer one
                                relevant_ios[2] = io
                            end
                            should_consider = false  # Don't push again, we either replaced or skipped
                        else
                            # No secondary IO yet, decide if we should add this one
                            if x_to_secondary < x_gap *0.3
                                should_consider = true
                            else
                                # Large gap: only consider if secondary is significantly closer
                                should_consider = x_to_secondary < x_to_primary
                            end
                        end
                    end
                    
                    if should_consider
                        push!(relevant_ios, io)
                    end
                end
            end
            
            if !isempty(relevant_ios)
                item_to_ios[item_id] = relevant_ios
            end
        end

        # LM only: honour per-item IO commitments. Prune stale ones (delivered
        # items are gone from `items`), then force each still-committed item's
        # target IO to the front of its candidate list so pass 1 serves it toward
        # the IO it's already been moving toward — no mid-delivery target flip.
        if UNIT_STEP[]
            for k in collect(keys(COMMITTED_IO))
                haskey(items, k) || delete!(COMMITTED_IO, k)
            end
            for (item_id, cio) in COMMITTED_IO
                cand = get(item_to_ios, item_id, Tuple[])
                cand = Tuple[io for io in cand if Tuple(io) != cio]
                pushfirst!(cand, cio)
                item_to_ios[item_id] = length(cand) > 2 ? cand[1:2] : cand
            end
        end

        # Single shared blockmat and escort availability across all IOs
        availableescorts = deepcopy(collect(keys(escorts)))
        blockmat = [0 for _ in 1:size(matrix, 1), _ in 1:size(matrix, 2)]
      #=   if iteration==3
            println("Item to IO mapping: ")
        end  =#
        # An item can appear in item_to_ios for up to 2 IOs (primary = closest, at index 1;
        # secondary = fallback candidate at index 2, if any). item.direction is a single
        # shared field representing the item's one physical move this iteration, so it must
        # be assigned an escort for at most one IO. To respect the primary/secondary priority
        # (rather than an arbitrary "whichever IO is processed first in the global loop"
        # order), this runs in two passes: pass 1 tries only each item's primary IO; pass 2
        # picks up leftover items (primary pass found no escort) via their secondary IO.
        already_assigned = Set{String}()

        # Shared per-IO assignment body, used by both passes below.
        function assign_for_io!(io, relevant_items)
            # LM mode with a single escort: serve the nearest item first, not the
            # farthest, so the lone escort doesn't abandon an adjacent load to chase a
            # distant one. With >1 escort the "farthest first" order load-balances
            # better (far escort commits to the far item), so keep it there.
            sorted_keys = (ESCORT_PROXIMITY_ORDER[] && UNIT_STEP[]) ?
                sort_keys_by_escort_proximity(items, escorts, availableescorts, relevant_items) :
                sort_keys_by_distance_and_sum(items, io, relevant_items;
                                              nearest_first = UNIT_STEP[] && length(escorts) == 1)

            if DBG_IOASSIGN[] != "" && any(w -> w in relevant_items, split(DBG_IOASSIGN[]))
                println("  [ASSIGN io=$io t=$iteration] order=$(sorted_keys)")
                for w in split(DBG_IOASSIGN[])
                    haskey(items, w) && println("      $w coords=$(items[w].coords) dir=$(items[w].direction)  primary_io=$(get(item_to_ios, w, "?"))")
                    haskey(escorts, w) && println("      $w(esc) coords=$(escorts[w].coords) available=$(w in availableescorts)")
                end
            end

            itemescortdict = Dict{String, Tuple{Vector{String}, Vector{String}}}()
            escort_items_dict = Dict{String, Tuple{Vector{String}, Vector{String}}}()

            updateitemescorts!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat, iteration, io)

            if DBG_IOASSIGN[] != "" && any(w -> w in relevant_items, split(DBG_IOASSIGN[]))
                for w in split(DBG_IOASSIGN[])
                    haskey(itemescortdict, w) && println("      $w  idle=$(get(ITEM_IDLE,w,0)) tol=$(idle_tol(w))  candX=$(itemescortdict[w][1])  candY=$(itemescortdict[w][2])")
                end
            end

            # Assignment loop for this specific IO - use stable iteration
            remaining_keys = deepcopy(sorted_keys)
            processing_order = use_zigzag ? zigzag_interleave(deepcopy(sorted_keys)) : deepcopy(sorted_keys)
            for key in processing_order
                if !haskey(itemescortdict, key)
                    continue  # already assigned as a doubleserve
                end
                item = items[key]
                escortsx = itemescortdict[key][1]
                escortsy = itemescortdict[key][2]
                x , y = items[key].coords
                filter!(x -> x != key, remaining_keys) # remove this item from remaining

                if (length(escortsx) == 0 && length(escortsy) == 0)
                    item.direction = 0 # not move
                    continue
                end

                escortid = ""  # Track the escort assigned

                if length(escortsx) == 0 && length(escortsy) > 0
                    item.direction = 2 # move in y
                    escortid = find_nearest_escort_multi_io(key, items, remaining_keys, matrix, io, blockmat, 2, escorts, escortsx, escortsy, iteration, IO, item_to_ios)
                    if escortid == ""
                        itemescortdict[key] = (itemescortdict[key][1], Vector{String}())
                        continue
                    else
                        updateblockmat!(blockmat, item, escorts[escortid])
                        filter!(x -> x != escortid, availableescorts)
                        updateitemescortslight!(itemescortdict, items, remaining_keys, escorts, availableescorts, blockmat, iteration, io)
                    end

                elseif length(escortsy) == 0 && length(escortsx) > 0
                    item.direction = 1
                    escortid = find_nearest_escort_multi_io(key, items, remaining_keys, matrix, io, blockmat, 1, escorts, escortsx, escortsy, iteration, IO, item_to_ios)
                    if escortid == ""
                        itemescortdict[key] = (Vector{String}(), itemescortdict[key][2])
                        continue
                    else
                        updateblockmat!(blockmat, item, escorts[escortid])
                        filter!(x -> x != escortid, availableescorts)
                        updateitemescortslight!(itemescortdict, items, remaining_keys, escorts, availableescorts, blockmat, iteration, io)
                    end

                elseif length(escortsx) > 0 && length(escortsy) > 0 # prefer x direction
                    preferred_dir = length(escortsx) > length(escortsy) ? 1 : 2
                    secondary_dir = preferred_dir == 1 ? 2 : 1
                    item.direction = preferred_dir
                    escortid = find_nearest_escort_multi_io(key, items, remaining_keys, matrix, io, blockmat, preferred_dir, escorts, escortsx, escortsy, iteration, IO, item_to_ios)
                    if escortid == ""
                        escortid = find_nearest_escort_multi_io(key, items, remaining_keys, matrix, io, blockmat, secondary_dir, escorts, escortsx, escortsy, iteration, IO, item_to_ios)
                        if escortid == ""
                            continue
                        else
                            item.direction = secondary_dir
                            updateblockmat!(blockmat, item, escorts[escortid])
                            filter!(x -> x != escortid, availableescorts)
                            updateitemescortslight!(itemescortdict, items, remaining_keys, escorts, availableescorts, blockmat, iteration, io)
                        end
                    else
                        updateblockmat!(blockmat, item, escorts[escortid])
                        filter!(x -> x != escortid, availableescorts)
                        updateitemescortslight!(itemescortdict, items, remaining_keys, escorts, availableescorts, blockmat, iteration, io)
                    end
                end

                if _dbg_io_watch(key) || (escortid != "" && _dbg_io_watch(escortid))
                    println("      -> $key (dir=$(item.direction)) got escort '$escortid'   io=$io  itemcoords=$((x,y))")
                end
                # Final assignment if escortid is not "" - save to escort_items_dict instead of modifying escorts directly
                if escortid != ""
                    if item.direction == 2
                        # Save itemsy assignment for this IO
                        if !haskey(escort_items_dict, escortid)
                            escort_items_dict[escortid] = (Vector{String}(), Vector{String}())
                        end
                        escort_items_dict[escortid] = (escort_items_dict[escortid][1], [key])
                    elseif item.direction == 1
                        # Save itemsx assignment for this IO
                        if !haskey(escort_items_dict, escortid)
                            escort_items_dict[escortid] = (Vector{String}(), Vector{String}())
                        end
                        escort_items_dict[escortid] = ([key], escort_items_dict[escortid][2])
                    end
                    push!(already_assigned, key)
                    # LM only: remember which IO this item is now being served toward,
                    # so next iteration it stays pinned to the same IO (see COMMITTED_IO).
                    UNIT_STEP[] && (COMMITTED_IO[key] = (io[1], io[2]))
                end

                # Clean up itemescortdict for items no longer being processed
                for rm_key in setdiff(collect(keys(itemescortdict)), remaining_keys)
                    delete!(itemescortdict, rm_key)
                end
                if all((length(itemescortdict[key][1]) + length(itemescortdict[key][2])) == 0 for key in keys(itemescortdict))
                    break
                end
            end

            return escort_items_dict
        end

        # Order the IOs for Pass 1. Default (IO_ORDER_MODE 0) keeps the IO
        # vector's own order (x-descending). LM only. Reordering matters because
        # the passes share one escort pool, so a later IO can be left without
        # escorts that an earlier IO already took.
        #   1 = most escorts in zone first    2 = fewest escorts in zone first
        #   3 = most items in zone first      4 = fewest items in zone first
        pass1_ios = IO
        if UNIT_STEP[] && IO_ORDER_MODE[] != 0
            m = IO_ORDER_MODE[]
            zonecount = Dict(io => 0 for io in IO)
            if m == 1 || m == 2
                for (_, e) in escorts
                    nio = argmin(io -> abs(io[1] - e.coords[1]) + abs(io[2] - e.coords[2]), IO)
                    zonecount[nio] += 1
                end
            else  # 3 or 4: count items by their primary IO
                for (k, _) in items
                    if haskey(item_to_ios, k) && !isempty(item_to_ios[k])
                        zonecount[item_to_ios[k][1]] += 1
                    end
                end
            end
            pass1_ios = sort(collect(IO), by = io -> zonecount[io], rev = (m == 1 || m == 3))
        end

        # Pass 1: each item tries its primary (closest) IO only
        for io in pass1_ios
            relevant_items = [item_id for item_id in keys(items)
                               if haskey(item_to_ios, item_id) && !isempty(item_to_ios[item_id]) && item_to_ios[item_id][1] == io]
            io_assignments[io] = isempty(relevant_items) ? Dict() : assign_for_io!(io, relevant_items)
        end

        # Pass 2: items whose primary pass found no escort fall back to their secondary IO
        for io in IO
            relevant_items = [item_id for item_id in keys(items)
                               if haskey(item_to_ios, item_id) && !(item_id in already_assigned) && io in item_to_ios[item_id]]
            if isempty(relevant_items)
                continue  # don't clobber pass-1 results for this io with an empty Dict
            end
            fallback_dict = assign_for_io!(io, relevant_items)
            existing = get(io_assignments, io, Dict())
            for (eid, (ix, iy)) in fallback_dict
                if haskey(existing, eid)
                    ex, ey = existing[eid]
                    existing[eid] = (vcat(ex, ix), vcat(ey, iy))
                else
                    existing[eid] = (ix, iy)
                end
            end
            io_assignments[io] = existing
        end

        # Build global_escort_items from all IO assignments
        global_escort_items = Dict{String, Vector{Tuple{Tuple, Vector{String}, Vector{String}}}}()

        for io in IO
            if haskey(io_assignments, io)
                escort_items_dict = io_assignments[io]
                for (escort_id, (itemsx, itemsy)) in escort_items_dict
                    if !haskey(global_escort_items, escort_id)
                        global_escort_items[escort_id] = Tuple{Tuple, Vector{String}, Vector{String}}[]
                    end
                    push!(global_escort_items[escort_id], (io, itemsx, itemsy))
                end
            end
        end

        # Idle-flex bookkeeping: an item counts as idling when its position has
        # not changed since the previous assignment call -- covers both "never
        # assigned" and "assigned but stuck".
        if IDLE_FLEX[]
            for k in keys(items)
                cur = items[k].coords
                ITEM_IDLE[k] = get(ITEM_LASTPOS, k, nothing) == cur ? get(ITEM_IDLE, k, 0) + 1 : 0
                ITEM_LASTPOS[k] = cur
            end
        end

        return (blockmat, global_escort_items, item_to_ios)

    else
        throw(ArgumentError("IO in wrong format: should be a Tuple{x=int,y=int} or an Vector{Tuple{Int,Int}}"))
    end
end

"""
GRASP version of item_escort_IO_assigment! — identical except items are visited
in a randomized (GRASP, α=1.0) order per IO instead of strict distance+escort-count
order, matching the same swap used in item_escort_assigment_r! for the single-IO case.
find_nearest_escort_multi_io itself stays deterministic (nearest wins), same as the
single-IO find_nearest_escort is left unrandomized in item_escort_assigment_r!.
"""
function item_escort_IO_assigment_r!(matrix, items, escorts, iteration, IO)
    if isa(IO, Vector{Tuple{Int,Int}})
        io_assignments = Dict()  # Will store: IO => (escortstomovefirst, blockmat, itemescortdict)

        resetescorts!(escorts, iteration)  # Reset escorts once for all IOs

        # Pre-process: determine which IOs are relevant for each item
        item_to_ios = Dict{String, Vector{Tuple}}()  # item_id => [primary_io, secondary_io (if any)]

        for item_id in keys(items)
            item_x, item_y = items[item_id].coords

            # Calculate distances to all IOs
            io_distances = [(io, euclidean_distance((item_x, item_y), io)) for io in IO]
            sort!(io_distances, by = x -> x[2])  # Sort by distance

            # Keep at most 2 closest IOs, with x-coordinate filtering
            relevant_ios = Tuple{Int,Int}[]
            for (i, (io, dist)) in enumerate(io_distances[1:min(2, length(io_distances))])
                io_x, io_y = io
                should_consider = true

                # X-coordinate alignment check
                if i == 1
                    # Always consider the closest IO
                    push!(relevant_ios, io)
                else
                    # For secondary IO, check x-coordinate alignment
                    primary_io_x, _ = io_distances[1][1]

                    # Don't consider secondary IO if item is not between/beyond the IOs
                    if (item_x < min(primary_io_x, io_x)) || (item_x > max(primary_io_x, io_x))
                        # Item is on one side, only one IO is relevant
                        if item_x < min(primary_io_x, io_x)
                            # Item is to the left, keep the leftmost IO (already primary)
                            should_consider = false
                        elseif item_x > max(primary_io_x, io_x)
                            # Item is to the right, replace primary with rightmost if primary is not rightmost
                            if primary_io_x > io_x
                                should_consider = false
                            else
                                # Primary is leftmost, secondary is rightmost - keep both logic
                                push!(relevant_ios, io)
                                should_consider = false
                            end
                        end
                    else
                        # Item is between or near both IOs
                        x_gap = abs(primary_io_x - io_x)
                        x_to_primary = abs(item_x - primary_io_x)
                        x_to_secondary = abs(item_x - io_x)

                        # Check if we already have a secondary IO; if current candidate is closer, replace it
                        if length(relevant_ios) > 1
                            existing_io = relevant_ios[2]
                            existing_io_x, _ = existing_io

                            # If current candidate is closer to item than existing secondary, replace it
                            if x_to_secondary < abs(item_x - existing_io_x)
                                # Replace the secondary IO with the closer one
                                relevant_ios[2] = io
                            end
                            should_consider = false  # Don't push again, we either replaced or skipped
                        else
                            # No secondary IO yet, decide if we should add this one
                            if x_to_secondary < x_gap *0.3
                                should_consider = true
                            else
                                # Large gap: only consider if secondary is significantly closer
                                should_consider = x_to_secondary < x_to_primary
                            end
                        end
                    end

                    if should_consider
                        push!(relevant_ios, io)
                    end
                end
            end

            if !isempty(relevant_ios)
                item_to_ios[item_id] = relevant_ios
            end
        end

        # LM only: honour per-item IO commitments (see item_escort_IO_assigment!).
        if UNIT_STEP[]
            for k in collect(keys(COMMITTED_IO))
                haskey(items, k) || delete!(COMMITTED_IO, k)
            end
            for (item_id, cio) in COMMITTED_IO
                cand = get(item_to_ios, item_id, Tuple[])
                cand = Tuple[io for io in cand if Tuple(io) != cio]
                pushfirst!(cand, cio)
                item_to_ios[item_id] = length(cand) > 2 ? cand[1:2] : cand
            end
        end

        # Single shared blockmat and escort availability across all IOs
        availableescorts = deepcopy(collect(keys(escorts)))
        blockmat = [0 for _ in 1:size(matrix, 1), _ in 1:size(matrix, 2)]
        # See item_escort_IO_assigment! for why this is needed: an item relevant to 2 IOs
        # must not be assigned twice across passes, or item.direction gets silently
        # overwritten while the earlier (now stale) assignment lingers in global_escort_items.
        # Two passes respect the primary/secondary priority in item_to_ios (index 1 = closest)
        # instead of an arbitrary "whichever IO is processed first" order.
        already_assigned = Set{String}()

        # Shared per-IO assignment body, used by both passes below.
        function assign_for_io!(io, relevant_items)
            # RANDOMIZATION: GRASP sort (α=0.7) instead of the deterministic sort
            sorted_keys = sort_keys_by_distance_and_sum_grasp(items, io, 0.7; relevant_items=relevant_items)

            itemescortdict = Dict{String, Tuple{Vector{String}, Vector{String}}}()
            escort_items_dict = Dict{String, Tuple{Vector{String}, Vector{String}}}()

            updateitemescorts!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat, iteration, io)

            if DBG_IOASSIGN[] != "" && any(w -> w in relevant_items, split(DBG_IOASSIGN[]))
                for w in split(DBG_IOASSIGN[])
                    haskey(itemescortdict, w) && println("      $w  idle=$(get(ITEM_IDLE,w,0)) tol=$(idle_tol(w))  candX=$(itemescortdict[w][1])  candY=$(itemescortdict[w][2])")
                end
            end

            # Assignment loop for this specific IO - use stable iteration
            remaining_keys = deepcopy(sorted_keys)
            for key in deepcopy(sorted_keys)
                if !haskey(itemescortdict, key)
                    continue  # already assigned as a doubleserve
                end
                item = items[key]
                escortsx = itemescortdict[key][1]
                escortsy = itemescortdict[key][2]
                x , y = items[key].coords
                filter!(x -> x != key, remaining_keys) # remove this item from remaining

                if (length(escortsx) == 0 && length(escortsy) == 0)
                    item.direction = 0 # not move
                    continue
                end

                escortid = ""  # Track the escort assigned

                if length(escortsx) == 0 && length(escortsy) > 0
                    item.direction = 2 # move in y
                    escortid = find_nearest_escort_multi_io(key, items, remaining_keys, matrix, io, blockmat, 2, escorts, escortsx, escortsy, iteration, IO, item_to_ios)
                    if escortid == ""
                        itemescortdict[key] = (itemescortdict[key][1], Vector{String}())
                        continue
                    else
                        updateblockmat!(blockmat, item, escorts[escortid])
                        filter!(x -> x != escortid, availableescorts)
                        updateitemescortslight!(itemescortdict, items, remaining_keys, escorts, availableescorts, blockmat, iteration, io)
                    end

                elseif length(escortsy) == 0 && length(escortsx) > 0
                    item.direction = 1
                    escortid = find_nearest_escort_multi_io(key, items, remaining_keys, matrix, io, blockmat, 1, escorts, escortsx, escortsy, iteration, IO, item_to_ios)
                    if escortid == ""
                        itemescortdict[key] = (Vector{String}(), itemescortdict[key][2])
                        continue
                    else
                        updateblockmat!(blockmat, item, escorts[escortid])
                        filter!(x -> x != escortid, availableescorts)
                        updateitemescortslight!(itemescortdict, items, remaining_keys, escorts, availableescorts, blockmat, iteration, io)
                    end

                elseif length(escortsx) > 0 && length(escortsy) > 0 # prefer x direction
                    preferred_dir = length(escortsx) > length(escortsy) ? 1 : 2
                    secondary_dir = preferred_dir == 1 ? 2 : 1
                    item.direction = preferred_dir
                    escortid = find_nearest_escort_multi_io(key, items, remaining_keys, matrix, io, blockmat, preferred_dir, escorts, escortsx, escortsy, iteration, IO, item_to_ios)
                    if escortid == ""
                        escortid = find_nearest_escort_multi_io(key, items, remaining_keys, matrix, io, blockmat, secondary_dir, escorts, escortsx, escortsy, iteration, IO, item_to_ios)
                        if escortid == ""
                            continue
                        else
                            item.direction = secondary_dir
                            updateblockmat!(blockmat, item, escorts[escortid])
                            filter!(x -> x != escortid, availableescorts)
                            updateitemescortslight!(itemescortdict, items, remaining_keys, escorts, availableescorts, blockmat, iteration, io)
                        end
                    else
                        updateblockmat!(blockmat, item, escorts[escortid])
                        filter!(x -> x != escortid, availableescorts)
                        updateitemescortslight!(itemescortdict, items, remaining_keys, escorts, availableescorts, blockmat, iteration, io)
                    end
                end

                if _dbg_io_watch(key) || (escortid != "" && _dbg_io_watch(escortid))
                    println("      -> $key (dir=$(item.direction)) got escort '$escortid'   io=$io  itemcoords=$((x,y))")
                end
                # Final assignment if escortid is not "" - save to escort_items_dict instead of modifying escorts directly
                if escortid != ""
                    if item.direction == 2
                        # Save itemsy assignment for this IO
                        if !haskey(escort_items_dict, escortid)
                            escort_items_dict[escortid] = (Vector{String}(), Vector{String}())
                        end
                        escort_items_dict[escortid] = (escort_items_dict[escortid][1], [key])
                    elseif item.direction == 1
                        # Save itemsx assignment for this IO
                        if !haskey(escort_items_dict, escortid)
                            escort_items_dict[escortid] = (Vector{String}(), Vector{String}())
                        end
                        escort_items_dict[escortid] = ([key], escort_items_dict[escortid][2])
                    end
                    push!(already_assigned, key)
                    # LM only: remember which IO this item is now being served toward,
                    # so next iteration it stays pinned to the same IO (see COMMITTED_IO).
                    UNIT_STEP[] && (COMMITTED_IO[key] = (io[1], io[2]))
                end

                # Clean up itemescortdict for items no longer being processed
                for rm_key in setdiff(collect(keys(itemescortdict)), remaining_keys)
                    delete!(itemescortdict, rm_key)
                end
                if all((length(itemescortdict[key][1]) + length(itemescortdict[key][2])) == 0 for key in keys(itemescortdict))
                    break
                end
            end

            return escort_items_dict
        end

        # Order the IOs for Pass 1. Default (IO_ORDER_MODE 0) keeps the IO
        # vector's own order (x-descending). LM only. Reordering matters because
        # the passes share one escort pool, so a later IO can be left without
        # escorts that an earlier IO already took.
        #   1 = most escorts in zone first    2 = fewest escorts in zone first
        #   3 = most items in zone first      4 = fewest items in zone first
        pass1_ios = IO
        if UNIT_STEP[] && IO_ORDER_MODE[] != 0
            m = IO_ORDER_MODE[]
            zonecount = Dict(io => 0 for io in IO)
            if m == 1 || m == 2
                for (_, e) in escorts
                    nio = argmin(io -> abs(io[1] - e.coords[1]) + abs(io[2] - e.coords[2]), IO)
                    zonecount[nio] += 1
                end
            else  # 3 or 4: count items by their primary IO
                for (k, _) in items
                    if haskey(item_to_ios, k) && !isempty(item_to_ios[k])
                        zonecount[item_to_ios[k][1]] += 1
                    end
                end
            end
            pass1_ios = sort(collect(IO), by = io -> zonecount[io], rev = (m == 1 || m == 3))
        end

        # Pass 1: each item tries its primary (closest) IO only
        for io in pass1_ios
            relevant_items = [item_id for item_id in keys(items)
                               if haskey(item_to_ios, item_id) && !isempty(item_to_ios[item_id]) && item_to_ios[item_id][1] == io]
            io_assignments[io] = isempty(relevant_items) ? Dict() : assign_for_io!(io, relevant_items)
        end

        # Pass 2: items whose primary pass found no escort fall back to their secondary IO
        for io in IO
            relevant_items = [item_id for item_id in keys(items)
                               if haskey(item_to_ios, item_id) && !(item_id in already_assigned) && io in item_to_ios[item_id]]
            if isempty(relevant_items)
                continue  # don't clobber pass-1 results for this io with an empty Dict
            end
            fallback_dict = assign_for_io!(io, relevant_items)
            existing = get(io_assignments, io, Dict())
            for (eid, (ix, iy)) in fallback_dict
                if haskey(existing, eid)
                    ex, ey = existing[eid]
                    existing[eid] = (vcat(ex, ix), vcat(ey, iy))
                else
                    existing[eid] = (ix, iy)
                end
            end
            io_assignments[io] = existing
        end

        # Build global_escort_items from all IO assignments
        global_escort_items = Dict{String, Vector{Tuple{Tuple, Vector{String}, Vector{String}}}}()

        for io in IO
            if haskey(io_assignments, io)
                escort_items_dict = io_assignments[io]
                for (escort_id, (itemsx, itemsy)) in escort_items_dict
                    if !haskey(global_escort_items, escort_id)
                        global_escort_items[escort_id] = Tuple{Tuple, Vector{String}, Vector{String}}[]
                    end
                    push!(global_escort_items[escort_id], (io, itemsx, itemsy))
                end
            end
        end

        # Idle-flex bookkeeping: an item counts as idling when its position has
        # not changed since the previous assignment call -- covers both "never
        # assigned" and "assigned but stuck".
        if IDLE_FLEX[]
            for k in keys(items)
                cur = items[k].coords
                ITEM_IDLE[k] = get(ITEM_LASTPOS, k, nothing) == cur ? get(ITEM_IDLE, k, 0) + 1 : 0
                ITEM_LASTPOS[k] = cur
            end
        end

        return (blockmat, global_escort_items, item_to_ios)

    else
        throw(ArgumentError("IO in wrong format: should be a Tuple{x=int,y=int} or an Vector{Tuple{Int,Int}}"))
    end
end

"""
Interleaves a far-first-sorted key list into far,near,far,near,... order: take
alternately from the front (farthest remaining) and back (nearest remaining).
Gives near items regular turns without fully reversing to close-first (which
starves far items and destabilizes makespan). Validated over 1500 instances:
improves both makespan and flowtime vs. strict far-first, including for
left/corner-positioned IOs.
"""
function zigzag_interleave(sorted_keys)
    result = similar(sorted_keys)
    lo, hi = 1, length(sorted_keys)
    i = 1
    take_front = true
    while lo <= hi
        if take_front
            result[i] = sorted_keys[lo]; lo += 1
        else
            result[i] = sorted_keys[hi]; hi -= 1
        end
        take_front = !take_front
        i += 1
    end
    return result
end

function item_escort_assigment!(matrix, items, escorts, iteration, IO; use_zigzag=ZIGZAG_INTERLEAVE[])
    #save_item_escorts!(matrix, items, escorts, IO)
    io_x, io_y = IO
    sorted_keys = sort_keys_by_distance_and_sum(items, IO)
    escortstomovefirst= String[]
    # sort the items by the increasing number of total escorts
    resetescorts!(escorts, iteration)
    availableescorts = deepcopy(collect(keys(escorts)))
    itemescortdict = Dict{String, Tuple{Vector{String}, Vector{String}}}() # save number of escorts that can serve item
    blockmat = [0 for _ in 1:size(matrix, 1), _ in 1:size(matrix, 2)]
    # updateitemescorts! resets item.direction = 0 for every item below, so
    # snapshot the previous iteration's direction here (used by LM mode's
    # sticky-direction preference further down) before it's wiped.
    prev_directions = Dict(key => items[key].direction for key in sorted_keys)
    updateitemescorts!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat, iteration, IO)

    processing_order = use_zigzag ? zigzag_interleave(deepcopy(sorted_keys)) : sorted_keys
    for key in processing_order
        item = items[key]
        escortsx = itemescortdict[key][1]
        escortsy = itemescortdict[key][2]
        x , y = items[key].coords    
        sorted_keys = filter(x -> x != key, sorted_keys) # remove this item as now we will decide its future
        if length(escortsx) == 0 && length(escortsy) == 0
            item.direction = 0 # not move               
            continue
        end
        if length(escortsx) == 0 && length(escortsy) > 0
            item.direction = 2 # move in y
            escortid = find_nearest_escort(key, items, sorted_keys, matrix, IO, blockmat,2,escorts, escortsx, escortsy,iteration) # is 0 if no escort is available (path blocked)
            if escortid == ""
                #println("No escort found for item ", key)
                itemescortdict[key] = (itemescortdict[key][1], Vector{String}())
                continue
                
            else
                updateblockmat!( blockmat, item, escorts[escortid])
                filter!(x -> x != escortid, availableescorts)
                updateitemescortslight!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat,iteration, IO)
            end
        
        elseif length(escortsy) == 0 && length(escortsx) > 0
            item.direction = 1
            escortid = find_nearest_escort(key, items, sorted_keys, matrix, IO, blockmat,1,escorts, escortsx, escortsy, iteration) 
            if escortid == ""
                #println("No escort found for item ", key)
                itemescortdict[key] = (Vector{String}(), itemescortdict[key][2])
                continue
            else
                updateblockmat!( blockmat, item, escorts[escortid])
                filter!(x -> x != escortid, availableescorts)
                updateitemescortslight!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat, iteration,IO)
            end
        
        elseif length(escortsx)>0 && length(escortsy) > 0 # prefer x direction
            # LM mode: once an item has started moving along one axis, keep
            # pushing that same axis (instead of re-picking by candidate count
            # every iteration) until it genuinely runs out of candidates --
            # axis-count-based preferred_dir can flip 1<->2 almost every
            # 1-cell step as escorts drift in/out of alignment, forcing a
            # fresh escort handoff each time instead of one escort finishing
            # the axis it is already on.
            prevdir = prev_directions[key]
            preferred_dir = if UNIT_STEP[] && prevdir in (1, 2) &&
                                ((prevdir == 1 && !isempty(escortsx)) || (prevdir == 2 && !isempty(escortsy)))
                prevdir
            else
                length(escortsx) > length(escortsy) ? 1 : 2
            end
            secondary_dir = preferred_dir == 1 ? 2 : 1
            item.direction = preferred_dir
            escortid = find_nearest_escort(key, items, sorted_keys, matrix, IO, blockmat, preferred_dir, escorts,escortsx, escortsy, iteration)
            if escortid == ""
                escortid = find_nearest_escort(key, items, sorted_keys, matrix, IO, blockmat, secondary_dir, escorts, escortsx, escortsy,iteration)
                if escortid == ""
                    continue
                else
                    item.direction = secondary_dir
                    updateblockmat!(blockmat, item, escorts[escortid])
                    filter!(x -> x != escortid, availableescorts)
                    updateitemescortslight!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat,iteration, IO)
                end
            else
                updateblockmat!(blockmat, item, escorts[escortid])
                filter!(x -> x != escortid, availableescorts)
                updateitemescortslight!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat, iteration, IO)
            end
        end
        if escortid != "" && (item.direction == 1 || item.direction == 2)
            push!(escortstomovefirst, escortid)
            if item.direction == 2
                escorts[escortid].itemsy = [key]
            elseif item.direction == 1
                escorts[escortid].itemsx = [key]
            end
        end

        # Remove the key from sorted_keys for the next iteration and re sort according to number of escorts

        sorted_keys = sort_keys_by_distance_and_sum(items, IO)
        for key in setdiff(collect(keys(itemescortdict)), sorted_keys)
            delete!(itemescortdict, key)
        end
        if all((length(itemescortdict[key][1]) + length(itemescortdict[key][2])) == 0 for key in keys(itemescortdict))
            break
        end
    end
    #print_matrix(matrix, blockmat)
    return escortstomovefirst, blockmat

end
function item_escort_assigment_r!(matrix, items, escorts, iteration, IO) 
    #save_item_escorts!(matrix, items, escorts, IO)
    io_x, io_y = IO
    sorted_keys = sort_keys_by_distance_and_sum_grasp(items, IO, GRASP_ITEM_ALPHA[])
    escortstomovefirst= String[]
    # sort the items by the increasing number of total escorts
    resetescorts!(escorts, iteration)
    availableescorts = deepcopy(collect(keys(escorts)))
    itemescortdict = Dict{String, Tuple{Vector{String}, Vector{String}}}() # save number of escorts that can serve item
    blockmat = [0 for _ in 1:size(matrix, 1), _ in 1:size(matrix, 2)]
    updateitemescorts!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat, iteration, IO)
      
    for key in deepcopy(sorted_keys)
        item = items[key]
        escortsx = itemescortdict[key][1]
        escortsy = itemescortdict[key][2]
        x , y = items[key].coords    
        sorted_keys = filter(x -> x != key, sorted_keys) # remove this item as now we will decide its future
        if length(escortsx) == 0 && length(escortsy) == 0
            item.direction = 0 # not move               
            continue
        end
        if length(escortsx) == 0 && length(escortsy) > 0
            item.direction = 2 # move in y
            escortid = find_nearest_escort(key, items, sorted_keys, matrix, IO, blockmat,2,escorts, escortsx, escortsy,iteration) # is 0 if no escort is available (path blocked)
            if escortid == ""
                #println("No escort found for item ", key)
                itemescortdict[key] = (itemescortdict[key][1], Vector{String}())
                continue
                
            else
                updateblockmat!( blockmat, item, escorts[escortid])
                filter!(x -> x != escortid, availableescorts)
                updateitemescortslight!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat,iteration, IO)
            end
        
        elseif length(escortsy) == 0 && length(escortsx) > 0
            item.direction = 1
            escortid = find_nearest_escort(key, items, sorted_keys, matrix, IO, blockmat,1,escorts, escortsx, escortsy, iteration) 
            if escortid == ""
                #println("No escort found for item ", key)
                itemescortdict[key] = (Vector{String}(), itemescortdict[key][2])
                continue
            else
                updateblockmat!( blockmat, item, escorts[escortid])
                filter!(x -> x != escortid, availableescorts)
                updateitemescortslight!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat, iteration,IO)
            end
        
        elseif length(escortsx)>0 && length(escortsy) > 0 # prefer x direction
            preferred_dir = length(escortsx) > length(escortsy) ? 1 : 2
            secondary_dir = preferred_dir == 1 ? 2 : 1
            item.direction = preferred_dir
            escortid = find_nearest_escort(key, items, sorted_keys, matrix, IO, blockmat, preferred_dir, escorts,escortsx, escortsy, iteration)
            if escortid == ""
                escortid = find_nearest_escort(key, items, sorted_keys, matrix, IO, blockmat, secondary_dir, escorts, escortsx, escortsy,iteration)
                if escortid == ""
                    continue
                else
                    item.direction = secondary_dir
                    updateblockmat!(blockmat, item, escorts[escortid])
                    filter!(x -> x != escortid, availableescorts)
                    updateitemescortslight!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat,iteration, IO)
                end
            else
                updateblockmat!(blockmat, item, escorts[escortid])
                filter!(x -> x != escortid, availableescorts)
                updateitemescortslight!(itemescortdict, items, sorted_keys, escorts, availableescorts, blockmat, iteration, IO)
            end
        end
        if escortid != "" && (item.direction == 1 || item.direction == 2)
            push!(escortstomovefirst, escortid)
            if item.direction == 2
                escorts[escortid].itemsy = [key]
            elseif item.direction == 1
                escorts[escortid].itemsx = [key]
            end
        end

        # Remove the key from sorted_keys for the next iteration and re sort according to number of escorts

        sorted_keys = sort_keys_by_distance_and_sum_grasp(items, IO, GRASP_ITEM_ALPHA[])
        for key in setdiff(collect(keys(itemescortdict)), sorted_keys)
            delete!(itemescortdict, key)
        end
        if all((length(itemescortdict[key][1]) + length(itemescortdict[key][2])) == 0 for key in keys(itemescortdict))
            break
        end
    end
    #print_matrix(matrix, blockmat)
    # Defensive cleanup + diagnostics: remove any escort with empty item assignment
    filter!(eid -> begin
        ok = !isempty(escorts[eid].itemsx) || !isempty(escorts[eid].itemsy)
        if !ok
            println("BUG in item_escort_assigment_r!: escort $eid in escortstomovefirst but has empty itemsx AND itemsy — removing from movers")
        end
        ok
    end, escortstomovefirst)
    return escortstomovefirst, blockmat

end
function updateitemescorts!(itemescortdict, items, sorted_keys,  escorts, availableescorts, blockmat, iteration,  IO)
    io_x, io_y = IO
    for key in sorted_keys
        item = items[key]
        item.direction = 0
        escortsx = Vector{String}()
        escortsy = Vector{String}()
        x , y = items[key].coords
        tol = idle_tol(key)   # 0 normally; grows once the item has idled a while
        for escort_id in availableescorts
            ex, ey = escorts[escort_id].coords
            if abs(ey - y) <= tol && x != io_x
                if allowedOrder(ex, io_x,x) && # escort is on the right side and item has to move right
                    !hurts_load_across_io(items, key, ex, ey, io_x, x) &&
                    noblock(blockmat, x, y, ex, ey) &&
                    !(haskey(escorts[escort_id].banset, iteration) && key in escorts[escort_id].banset[iteration])
                    push!(escortsx, escort_id)
                end
            elseif abs(ex - x) <= tol && ey < y && # escort is on the right side and item has to move right
                noblock(blockmat, x, y, ex, ey) &&
                !(haskey(escorts[escort_id].banset, iteration) && key in escorts[escort_id].banset[iteration])# escort is below the item and the x coord is the save_item_escorts
                push!(escortsy, escort_id)
            end
        end
        itemescortdict[key] = (escortsx, escortsy)
        items[key].escortssum = length(escortsx) + length(escortsy)
    end
end
function updateitemescortslight!(itemescortdict, items, sorted_keys,  escorts, availableescorts, blockmat, iteration,  IO)
    io_x, io_y = IO
    # Remove keys from itemescortdict that are not in sorted_keys
    for key in sorted_keys
        item = items[key]
        escortsx = itemescortdict[key][1]
        escortsy = itemescortdict[key][2]
        x , y = items[key].coords
        for escort_id in escortsx
            ex, ey = escorts[escort_id].coords
            if !(ey == y && allowedOrder(ex, io_x,x) && # escort is on the right side and item has to move right
                !hurts_load_across_io(items, key, ex, ey, io_x, x) &&
                noblock(blockmat, x, y, ex, ey) && escort_id in availableescorts) || (haskey(escorts[escort_id].banset, iteration) && key in escorts[escort_id].banset[iteration])
                filter!(x -> x != escort_id, escortsx)
            end
        end
        for escort_id in escortsy
            ex, ey = escorts[escort_id].coords
            if !(ex == x && ey < y && 
                noblock(blockmat, x, y, ex, ey) && escort_id in availableescorts) || (haskey(escorts[escort_id].banset, iteration) && key in escorts[escort_id].banset[iteration])# escort is below the item and the x coord is the save_item_escorts
                filter!(x -> x != escort_id, escortsy)
            end
        end
        itemescortdict[key] = (escortsx, escortsy)
        items[key].escortssum = length(escortsx) + length(escortsy)
    end
end
function noblock(blockmat, x, y, ex, ey)
    if x == ex
        ystart = min(y, ey)
        yend = max(y, ey)
        for y in ystart:yend
            if blockmat[x, y] == 1
                return false
            end
        end
    elseif y == ey
        xstart = min(x, ex)
        xend = max(x, ex)
        for x in xstart:xend
            if blockmat[x, y] == 1
                return false
            end
        end
    end
    return true
end
function euclidean_distance(coords1, coords2)
    return sqrt((coords1[1] - coords2[1])^2 + (coords1[2] - coords2[2])^2)
end

# Order items by proximity of their nearest *available* escort: items with the
# most directly-adjacent (Manhattan <= 1) escorts first, then by increasing
# distance to their single nearest available escort, then by escortssum.
# Used by item_escort_IO_assigment! when ESCORT_PROXIMITY_ORDER[] is set.
function sort_keys_by_escort_proximity(items, escorts, availableescorts, relevant_items=nothing)
    keys_to_sort = relevant_items === nothing ? collect(keys(items)) :
                   intersect(relevant_items, collect(keys(items)))
    escoords = [escorts[e].coords for e in availableescorts if haskey(escorts, e)]
    function keyfun(x)
        ix, iy = items[x].coords
        isempty(escoords) && return (0, Inf, items[x].escortssum)
        dists = [abs(ex - ix) + abs(ey - iy) for (ex, ey) in escoords]
        (-count(<=(1), dists), minimum(dists), items[x].escortssum)
    end
    sort(keys_to_sort, by = keyfun)
end

"""
sorting function. By default: decreasing distance to IO, then increasing number of
escorts (furthest-away item first — gives the longest job a head start when many
escorts share the work). With nearest_first=true: closest item first — used in LM
mode, where a lone escort should finish the item it is already next to before
trekking to a far one (walking away from an adjacent item then coming back is the
main source of makespan bloat in the multi-IO lone-escort case).
"""
function sort_keys_by_distance_and_sum(items, IO, relevant_items=nothing; nearest_first::Bool=false)
    if relevant_items === nothing
        keys_to_sort = collect(keys(items))
    else
        keys_to_sort = intersect(relevant_items, collect(keys(items)))
    end
    s = nearest_first ? 1 : -1
    sorted_keys = sort(keys_to_sort, by = x -> (
        s * euclidean_distance(items[x].coords, IO),
        items[x].escortssum  # Sum of lengths for increasing order
    ))
    return sorted_keys
end

"""
Rank-biased GRASP pick: returns an index in 1:n with probability proportional to
α^(i-1), where i=1 is the best-ranked candidate (first in a best-to-worst sort).
α=1.0 → uniform over all n candidates. α→0 → collapses onto rank 1 (pure greedy).
Smaller α makes earlier (better) ranks disproportionately more likely.
"""
function grasp_rank_pick(rng, n::Int, α::Float64)
    n == 1 && return 1
    weights = α == 1.0 ? ones(n) : [α^(i-1) for i in 1:n]
    total = sum(weights)
    r = rand(rng) * total
    cum = 0.0
    for i in 1:n
        cum += weights[i]
        r <= cum && return i
    end
    return n
end

"""
GRASP version of sort_keys_by_distance_and_sum.
At each step, ranks the remaining items by distance to IO (furthest first) and picks
one with probability proportional to α^(rank-1) via grasp_rank_pick.
α=1.0 → uniform random choice; α→0 → always the furthest item (pure greedy).
"""
function sort_keys_by_distance_and_sum_grasp(items, IO, α::Float64; relevant_items=nothing)
    if relevant_items === nothing
        pool = collect(keys(items))
    else
        pool = intersect(relevant_items, collect(keys(items)))
    end

    result = String[]
    while !isempty(pool)
        # greedy metric: distance from IO (higher = better, we want furthest first)
        ranked = sort(pool, by = k -> -euclidean_distance(items[k].coords, IO))

        # RANDOMIZATION: probability of picking rank i is proportional to α^(i-1)
        idx = grasp_rank_pick(Random.default_rng(), length(ranked), α)
        chosen = ranked[idx]

        push!(result, chosen)
        filter!(k -> k != chosen, pool)
    end
    return result
end

"""
GRASP version of the non-movers sort used in moveescorts_flow!.
The greedy metric is distance_to_IO (furthest first). At each step, ranks the
remaining escorts by that metric and picks one with probability proportional to
α^(rank-1) via grasp_rank_pick.
α=1.0 → uniform random choice; α→0 → always the furthest escort (pure greedy).
"""
function sort_nonmovers_grasp(nonmovers, escorts, items, matrix, blockmat, IO, α::Float64)
    pool = collect(nonmovers)

    # Precompute the scalar key: distance to IO (we want largest first)
    function dist_key(escortid)
        esc_x, esc_y = escorts[escortid].coords
        return euclidean_distance((esc_x, esc_y), IO)
    end

    result = String[]
    while !isempty(pool)
        ranked = sort(pool, by = e -> -dist_key(e))

        # RANDOMIZATION: probability of picking rank i is proportional to α^(i-1)
        idx = grasp_rank_pick(Random.default_rng(), length(ranked), α)
        chosen = ranked[idx]

        push!(result, chosen)
        filter!(e -> e != chosen, pool)
    end
    return result
end

"""
Multi-IO version of sort_nonmovers_grasp — the greedy metric per escort is
distance_to_nearest_IO (furthest first), mirroring the deterministic multi-IO
nonmovers sort in moveescorts_flow_multi_io! but with a rank-biased random pick
instead of a plain sort.
α=1.0 → uniform random choice; α→0 → always the furthest escort (pure greedy).
"""
function sort_nonmovers_multi_io_grasp(nonmovers, escorts, items, matrix, blockmat, all_ios, α::Float64)
    pool = collect(nonmovers)

    function dist_key(escortid)
        esc_x, esc_y = escorts[escortid].coords
        return minimum(io -> abs(io[1] - esc_x) + abs(io[2] - esc_y), all_ios)
    end

    result = String[]
    while !isempty(pool)
        ranked = sort(pool, by = e -> -dist_key(e))

        # RANDOMIZATION: probability of picking rank i is proportional to α^(i-1)
        idx = grasp_rank_pick(Random.default_rng(), length(ranked), α)
        chosen = ranked[idx]

        push!(result, chosen)
        filter!(e -> e != chosen, pool)
    end
    return result
end

"""
GRASP version of the nearest-escort-to-its-item mover ordering used in
moveescorts_flow!. The greedy metric is distance from the escort to its own
target item (closest over itemsx ∪ itemsy — the item eventually served may be
randomly chosen from either list, so this is a proxy for "how easy this
escort's move is"), nearest first. At each step, ranks the remaining movers by
that metric and picks one with probability proportional to α^(rank-1) via
grasp_rank_pick.
α=1.0 → uniform random order; α→0 → always the nearest-first escort (pure
greedy, matching moveescorts_flow!'s deterministic order).
"""
function sort_movers_by_item_distance_grasp(moverescortids, escorts, items, α::Float64)
    function dist_key(escortid)
        esc_x, esc_y = escorts[escortid].coords
        candidates = vcat(escorts[escortid].itemsx, escorts[escortid].itemsy)
        isempty(candidates) && return Inf
        return minimum(k -> abs(esc_x - items[k].coords[1]) + abs(esc_y - items[k].coords[2]), candidates)
    end

    pool = collect(moverescortids)
    result = String[]
    while !isempty(pool)
        ranked = sort(pool, by = dist_key)

        # RANDOMIZATION: probability of picking rank i is proportional to α^(i-1)
        idx = grasp_rank_pick(Random.default_rng(), length(ranked), α)
        chosen = ranked[idx]

        push!(result, chosen)
        filter!(e -> e != chosen, pool)
    end
    return result
end

function sort_keys_by_distance(items, IO, increasing)
    if increasing
        sorted_keys = sort(collect(keys(items)), by = x -> (
        euclidean_distance(items[x].coords, IO))) # Negative Euclidean distance for decreasing order
    else
        sorted_keys = sort(collect(keys(items)), by = x -> (
            -euclidean_distance(items[x].coords, IO))) 
    end
    
    return sorted_keys
end

function sort_urgkeys_by_distance_toescort(items, urgkeys, esccoords,increasing)
    if increasing
        sorted_keys = sort(collect(urgkeys), by = x -> (
        euclidean_distance(items[x].coords, esccoords))) # Negative Euclidean distance for decreasing order
    else
        sorted_keys = sort(collect(urgkeys), by = x -> (
            -euclidean_distance(items[x].coords, esccoords)))
    end

    return sorted_keys
end

"""
GRASP version of sort_urgkeys_by_distance_toescort.
At each step, ranks the remaining urgent items by distance to the escort
(closest first if increasing, furthest first otherwise) and picks one with
probability proportional to α^(rank-1) via grasp_rank_pick.
α=1.0 → uniform random choice; α→0 → always the best-ranked item (pure greedy).
increasing=true  → closest items first (escort goes to nearest urgent item).
increasing=false → furthest items first.
"""
function sort_urgkeys_by_distance_toescort_grasp(items, urgkeys, esccoords, increasing, α::Float64)
    pool = collect(urgkeys)
    result = String[]

    while !isempty(pool)
        ranked = increasing ?
            sort(pool, by = k -> euclidean_distance(items[k].coords, esccoords)) :
            sort(pool, by = k -> -euclidean_distance(items[k].coords, esccoords))

        # RANDOMIZATION: probability of picking rank i is proportional to α^(i-1)
        idx = grasp_rank_pick(Random.default_rng(), length(ranked), α)
        chosen = ranked[idx]

        push!(result, chosen)
        filter!(k -> k != chosen, pool)
    end
    return result
end

"""
updateblockmat! updates the blockmat with the block between item and escort after assignment
"""
function updateblockmat!( blockmat, item, escort) # ban the block between item and escort
    itemx, itemy = item.coords
    escortx, escorty = escort.coords
    if UNIT_STEP[]
        # LM mode: this reserves the corridor between a just-assigned escort
        # and its item, but the escort will only take one physical step this
        # iteration (assignment is recomputed fresh every iteration anyway).
        # Claiming the whole span up front blocks other escorts from cells
        # the assigned escort will not actually reach for many iterations —
        # same phantom-blocked-corridor class as updateblockmat_e!.
        if itemx == escortx
            itemy = escorty + sign(itemy - escorty) * min(abs(itemy - escorty), 1)
        elseif itemy == escorty
            itemx = escortx + sign(itemx - escortx) * min(abs(itemx - escortx), 1)
        end
    end
    if  itemx == escortx
        ystart = min(itemy, escorty)
        yend = max(itemy, escorty)
        for y in ystart:yend
            blockmat[itemx, y] = 1
        end
    elseif itemy == escorty
        xstart = min(itemx, escortx)
        xend = max(itemx, escortx)
        for x in xstart:xend
            blockmat[x, itemy] = 1
        end
    end
end
"""
updateblockmat_e! updates the blockmat with the block between escort curr and escort fin coords
"""
function updateblockmat_e!( blockmat, escortx, escorty, finx, finy; val = 1) # ban the block between item and escort
    if UNIT_STEP[]
        # LM mode: a real move is never more than one cell, no matter what
        # (possibly uncapped) target a caller computed. Marking a
        # multi-cell span here would fabricate a "phantom blocked corridor"
        # over cells that were never actually traversed this iteration —
        # confirmed root cause of escorts seeing false path_blocked results
        # from other escorts' intended-but-LM-clamped moves.
        if escortx == finx
            finy = escorty + sign(finy - escorty) * min(abs(finy - escorty), 1)
        elseif escorty == finy
            finx = escortx + sign(finx - escortx) * min(abs(finx - escortx), 1)
        end
    end
    if escortx == finx # direction Y
        ystart = min(escorty, finy)
        yend = max(escorty, finy)
        for y in ystart:yend
            blockmat[escortx, y] = val
        end
    elseif escorty == finy # direction X
        xstart = min(escortx, finx)
        xend = max(escortx, finx)
        for x in xstart:xend
            blockmat[x, escorty] = val
        end
    end
end
function updateurgmats_e!(urgmats, escortx, escorty, finx, finy; val = 1)
    for urgid in keys(urgmats)
        if escortx == finx # direction Y 
            ystart = min(escorty, finy)
            yend = max(escorty, finy)
            for y in ystart:yend
                urgmats[urgid][escortx, y] = val
            end           
        elseif escorty == finy # direction X
            xstart = min(escortx, finx)
            xend = max(escortx, finx)
            for x in xstart:xend
                urgmats[urgid][x, escorty] = val
            end
        end
    end
end
"""
Given everything it finds the nearest escort to item. checks the path, if another item can be servd it serves it too.

"""
#=
# Multi-IO Find Nearest Escort Function - Logic Explanation

An escort should NOT be assigned to an item if:

1. **The escort is "too far out"**: The escort lies beyond another IO (from current IO perspective) that has unassigned items needing to move in the same direction
   - Example: `item1, io1, io2, escort, item2`
   - item1 moves toward io1, item2 moves toward io2
   - Escort is beyond io2 - should serve item2 for io2, not item1 for io1

2. **Multiple items same direction**: There are other unassigned items that also need to move in the same direction toward the same IO
   - Example: `item1, io1, item2, io2, escort`
   - Both items move right toward their respective IOs
   - Escort should wait to serve both (or the one further out first to maintain furthest-first principle)


# Algorithm Steps


# Step 2: Check escort distance vs other IOs
For each candidate escort:
- Calculate distance from escort to current_io
- For each other IO in `all_ios`:
  - Check if that IO has unassigned items needing same direction
  - If escort is FURTHER from current_io than that other IO is, SKIP escort
  - Reasoning: Escort should serve the IO it's closer to

# Step 3: Detect multi-item opportunity (doubleserve)
- Check path between item and escort
- Identify other items on this path that could be served together
- Mark them for batch assignment

# Step 4: Validate path and return
- Ensure path is not blocked
- Return escort ID if valid, 0 otherwise

# Key Variables

- `escort_dist_to_current_io`: Distance from escort to current IO
- `other_io_dist_to_current_io`: Distance from another IO to current IO
- `competing_items`: Items that also move in same direction to same IO

=#

function find_nearest_escort_multi_io(itemid::String,items::Dict,remaining_keys::Vector, matrix::Matrix, current_io::Tuple, blockmat::Matrix, direction::Int,
    escorts::Dict,relevantescx::Vector,relevantescy::Vector,    iteration::Int,    all_ios::Vector{Tuple{Int,Int}},    item_to_ios::Dict
)
    itemx, itemy = items[itemid].coords
    nearest_id = ""
    min_dist = Inf
    iox, ioy = current_io
    doubleserve = String[]
  #=   if(iteration ==3 && itemid == "I1")
        println("here")
    end  =#
    # Helper: check if escort would better serve another unassigned item
    function escort_better_for_other_item(escort_x, escort_y, direction) #TODO Monday
        # Calculate distance from current item to escort (only relevant coordinate)
        if direction == 1
            item_dist_to_escort = abs(escort_x - itemx)  # x-distance for x-movement
        else  # direction == 2
            item_dist_to_escort = abs(escort_y - itemy)  # y-distance for y-movement
        end
        
        # Check if an unassigned item closer to escort targets a different IO
        for candidate_key in (filter(k -> k != itemid, collect(keys(items))))
            if !haskey(item_to_ios, candidate_key)
                continue
            end
            
            candidate_x, candidate_y = items[candidate_key].coords
            
            # Calculate distance from candidate to escort (only relevant coordinate)
            if direction == 1
                candidate_dist_to_escort = abs(escort_x - candidate_x)
                # Is candidate CLOSER to escort than current item?
                if candidate_dist_to_escort < item_dist_to_escort && candidate_x == escort_x
                    candidate_ios = item_to_ios[candidate_key]
                    
                    # Check each IO the candidate targets
                    for candidate_io in candidate_ios
                        if candidate_io == current_io
                            continue
                        end
                        
                        # Case 1: itemx < current_io[1] < candidate_io[1] < candidate_x < escort_x
                        # Case 2: escort_x < candidate_x < candidate_io[1] < current_io[1] < itemx (mirrored)
                        if (itemx < current_io[1] < candidate_io[1] < candidate_x < escort_x ) ||
                            (escort_x < candidate_x < candidate_io[1] < current_io[1] < itemx)
                            return true
                        end
                       
                    end
                end
            elseif direction == 2
                
                candidate_dist_to_escort = abs(escort_y - candidate_y)
                # Is candidate CLOSER to escort than current item?
                if candidate_dist_to_escort < item_dist_to_escort && candidate_y == escort_y
                    candidate_ios = item_to_ios[candidate_key]
                    
                    # Check each IO the candidate targets
                    for candidate_io in candidate_ios
                        if candidate_io == current_io
                            continue
                        end
                        
                        # Not allowed cases (y-direction)
                        # Case 1: itemy < current_io[2] < candidate_io[2] < candidate_y < escort_y
                        # Case 2: escort_y < candidate_y < candidate_io[2] < current_io[2] < itemy (mirrored)
                        if (itemy < current_io[2] < candidate_io[2] < candidate_y < escort_y) ||
                            (escort_y < candidate_y < candidate_io[2] < current_io[2] < itemy)
                            return true
                        end
                        
                    end
                end
            end
        end
        return false
    end


    # Direction 1: x-direction movement
    if direction == 1
        
        for e_id in relevantescx
            escort_x, escort_y = escorts[e_id].coords
            
            # Ban check
            if haskey(escorts[e_id].banset, iteration) && itemid ∈ escorts[e_id].banset[iteration]
                continue
            end
            
            # Basic sanity checks
            if abs(escort_y - itemy) > idle_tol(itemid) || (iox < itemx && escort_x > itemx) || (iox > itemx && escort_x < itemx)
                continue
            end

            # MULTI-IO CONSTRAINT: Check if escort would better serve another unassigned item
            if escort_better_for_other_item(escort_x, escort_y, 1)
                continue
            end
            
            # Check blockmat between itemx and escort_x at y = itemy
            xstart = min(itemx, escort_x)
            xend = max(itemx, escort_x)
            path_blocked = false
            
            for xx in xstart:xend
                if blockmat[xx, itemy] == 1
                    path_blocked = true
                    break
                elseif matrix[xx, itemy] ∈ (filter(k -> k != itemid, collect(keys(items))))
                    if haskey(escorts[e_id].banset, iteration) && matrix[xx, itemy] ∈ escorts[e_id].banset[iteration]
                        path_blocked = true
                    elseif !allowedOrder(escort_x, iox, xx, itemx)
                        path_blocked = true
                    elseif matrix[xx, itemy] ∈ remaining_keys
                        push!(doubleserve, matrix[xx, itemy])
                    else
                        path_blocked = true  # item already assigned, can't pass through
                    end
                end
            end
            
            if path_blocked
                continue
            end
            
            dist = abs(escort_x - itemx)
            sticky = UNIT_STEP[] && get(COMMITTED_ESCORT, itemid, "") == e_id
            if dist < min_dist || (dist == min_dist && sticky)
                min_dist = dist
                nearest_id = e_id
            end
        end

    elseif direction == 2  # y-direction movement
        
        for e_id in relevantescy
            escort_x, escort_y = escorts[e_id].coords
            
            # Ban check
            if haskey(escorts[e_id].banset, iteration) && itemid ∈ escorts[e_id].banset[iteration]
                continue
            end
            
            # Basic sanity checks
            if abs(escort_x - itemx) > idle_tol(itemid) || escort_y > itemy
                continue
            end
            
            # MULTI-IO CONSTRAINT: Check if escort would better serve another unassigned item
            if escort_better_for_other_item(escort_x, escort_y, 2)
                continue
            end
        
            # Check blockmat between itemy and escort_y at x = itemx
            ystart = min(itemy, escort_y)
            yend = max(itemy, escort_y)
            path_blocked = false
            
            for yy in ystart:yend
                if blockmat[itemx, yy] == 1
                    path_blocked = true
                    break
                elseif matrix[itemx, yy] ∈ (filter(k -> k != itemid, collect(keys(items))))
                    if haskey(escorts[e_id].banset, iteration) && matrix[itemx, yy] ∈ escorts[e_id].banset[iteration]
                        path_blocked = true
                    elseif !allowedOrder(escort_y, ioy, yy, itemy)
                        path_blocked = true
                    elseif matrix[itemx, yy] ∈ remaining_keys
                        push!(doubleserve, matrix[itemx, yy])
                    else
                        path_blocked = true  # item already assigned, can't pass through
                    end
                end
            end
            
            if path_blocked
                continue
            end
            
            dist = abs(escort_y - itemy)
            sticky = UNIT_STEP[] && get(COMMITTED_ESCORT, itemid, "") == e_id
            if dist < min_dist || (dist == min_dist && sticky)
                min_dist = dist
                nearest_id = e_id
            end
        end
    end

    # LM only: remember which escort is currently serving this item so future
    # calls prefer it on ties instead of flip-flopping between equally-near
    # escorts every iteration (mirrors find_nearest_escort's single-IO
    # commitment). Naturally released once the item is delivered -- mode
    # "continue" removes delivered items from `items`, so a stale entry here
    # just never matches any future itemid and imposes no preference on the
    # escort's next assignment (proximity alone decides who gets it next).
    if UNIT_STEP[] && nearest_id != ""
        COMMITTED_ESCORT[itemid] = nearest_id
    end

    if nearest_id != "" #TODO should we check for all ios this block?
        escort_x, escort_y = escorts[nearest_id].coords
        if ((abs(iox - escort_x) + abs(ioy - escort_y)) <= length(keys(items))) # escort in close proximity to IO, therefore its move will be controlled
            futurecoords = generatefuturecoords_multi_io(items, escorts, direction, nearest_id, itemid, matrix, item_to_ios, current_io)
            samecoords = Tuple{Int,Int}[]
            if direction == 2
                samecoords = filter(x -> x[1] == itemx &&  x[2] >= min(itemy, escort_y) && x[2] <= max(itemy, escort_y), futurecoords)
            elseif direction ==1
                samecoords = filter(x -> x[2] == itemy &&  x[1] >= min(itemx, escort_x) && x[1] <= max(itemx, escort_x), futurecoords)
            end
            if !isempty(samecoords)
                minDist = minimum([abs(iox - coord[1]) + abs(ioy - coord[2]) for coord in samecoords])
                if minDist <= length(keys(items)) && # item far out from IO
                    !path_to_io_exists_if(matrix, futurecoords, current_io)   # check with A* if this movement would cause some stupid block
                    items[itemid].direction = 0
                    return ""
                end
            end
        end
    
        for key in doubleserve # we are lucky to serve two items 
            filter!(x -> x != key, remaining_keys)
            items[key].direction = direction
            # If path is blocked, revert assignment:
            if direction == 1
                push!(escorts[nearest_id].itemsx, key)
            elseif direction == 2
                push!(escorts[nearest_id].itemsy, key)
            end
        end
    end


    return nearest_id
end

function find_nearest_escort(itemid, items, sorted_keys, matrix, IO, blockmat, direction,escorts, relevantescx, relevantescy, iteration)
    itemx, itemy = items[itemid].coords
    nearest_id = ""
    min_dist = Inf
    iox, ioy = IO
    doubleserve = String[]

    if direction == 1 # x_
        for e_id in relevantescx
            escort_x, escort_y = escorts[e_id].coords 
            if haskey(escorts[e_id].banset, iteration) && itemid ∈ escorts[e_id].banset[iteration]
                continue
            end
            if escort_y != itemy || (iox < itemx && escort_x > itemx) || (iox > itemx && escort_x < itemx)
                println("saved escort$e_id: ($escort_x, $escort_y) wrong,notsameY item $(itemid) ($itemx, $itemy)")
               # print_matrix(matrix)
                continue
            end
            # Check blockmat between itemx and escort_x at y = itemy
            xstart = min(itemx, escort_x)
            xend   = max(itemx, escort_x)
            path_blocked = false
            for xx in xstart:xend
                # blockmat entry is a tuple, skip if first value is 1
                if blockmat[xx, itemy] == 1 # either already serving or another item on the path
                    path_blocked = true
                    break
                elseif matrix[xx, itemy] ∈ sorted_keys # we serve double item, delete from sorted_keys
                    if haskey(escorts[e_id].banset, iteration) && matrix[xx, itemy] ∈ escorts[e_id].banset[iteration]
                        path_blocked = true
                    elseif !allowedOrder(escort_x, iox, xx, itemx) 
                        path_blocked = true
                    else
                        push!(doubleserve, matrix[xx, itemy])
                    end
                end
            end
            if path_blocked
                continue
            end
            dist = abs(escort_x - itemx)
            sticky = UNIT_STEP[] && get(COMMITTED_ESCORT, itemid, "") == e_id
            # Sticky only breaks ties in favor of the already-committed escort;
            # it must not override a genuinely closer candidate (e.g. one that
            # just finished aligning onto the item's row/column this iteration)
            # -- that was forcing dist=-1 unconditionally, which kept a farther
            # escort committed even after a strictly closer one became eligible.
            if dist < min_dist || (dist == min_dist && sticky)
                min_dist = dist
                nearest_id = e_id
            end
        end
    elseif direction == 2 # y
        for e_id in relevantescy
            escort_x, escort_y = escorts[e_id].coords
            if haskey(escorts[e_id].banset, iteration) && itemid ∈ escorts[e_id].banset[iteration]
                continue
            end
            if escort_x != itemx || escort_y > itemy
                println("saved escort$e_id: ($escort_x, $escort_y) wrong,notsameX item $(itemid) ($itemx, $itemy)")
                #print_matrix(matrix)
                continue
            end

            # Check blockmat between itemy and escort_y at x = itemx
            ystart = min(itemy, escort_y)
            yend   = max(itemy, escort_y)
            path_blocked = false
            for yy in ystart:yend
                # blockmat entry is a tuple, skip if first value is 1
                if blockmat[itemx, yy] == 1 #
                    path_blocked = true
                    break
                elseif matrix[itemx, yy] ∈ sorted_keys # we serve double item, delete from sorted_keys
                    if haskey(escorts[e_id].banset, iteration) && matrix[itemx, yy] ∈ escorts[e_id].banset[iteration]
                        path_blocked = true
                    elseif !allowedOrder(escort_y, ioy, yy, itemy)
                        println("IO_y in the path of y movement, should not happen, check error")
                        println(e_id, " ", item.id)
                        print_matrix(matrix)
                    else
                        push!(doubleserve, matrix[itemx, yy])
                    end
                end
            end
            if path_blocked
                continue
            end            
            dist = abs(escort_y - itemy)
            sticky = UNIT_STEP[] && get(COMMITTED_ESCORT, itemid, "") == e_id
            if dist < min_dist || (dist == min_dist && sticky)
                min_dist = dist
                nearest_id = e_id
            end
        end
    end
    if UNIT_STEP[] && nearest_id != ""
        COMMITTED_ESCORT[itemid] = nearest_id
    end
    if nearest_id != "" && !UNIT_STEP[]
        # This bail-out judges the uncapped, potentially multi-cell future
        # position (generatefuturecoords simulates the full BM-style push) —
        # under LM mode the escort only ever takes a single-cell step, so a
        # perfectly good adjacent escort candidate can get rejected (and the
        # item forced to direction=0) based on a jump that was never going to
        # happen. Skip this check entirely in LM mode.
        escort_x, escort_y = escorts[nearest_id].coords
        if ((abs(IO[1] - escort_x) + abs(IO[2] - escort_y)) <= length(keys(items))) # escort in close proximity to IO, therefore its move will be controlled
            futurecoords = generatefuturecoords(items, escorts,direction, nearest_id, itemid, matrix, IO)
            samecoords = Tuple{Int,Int}[]
            if direction == 2
                samecoords = filter(x -> x[1] == itemx &&  x[2] >= min(itemy, escort_y) && x[2] <= max(itemy, escort_y), futurecoords)
            elseif direction ==1
                samecoords = filter(x -> x[2] == itemy &&  x[1] >= min(itemx, escort_x) && x[1] <= max(itemx, escort_x), futurecoords)
            end
            if !isempty(samecoords)
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in samecoords])
                if minDist <= length(keys(items)) && # item far out from IO
                    !path_to_io_exists_if(matrix, futurecoords, IO)   # check with A* if this movement would cause some stupid block
                    items[itemid].direction = 0
                    return ""
                end
            end
        end
    
        for key in doubleserve # we are lucky to serve two items 
            filter!(x -> x != key, sorted_keys)
            items[key].direction = direction
            # If path is blocked, revert assignment:
            if direction == 1
                push!(escorts[nearest_id].itemsx, key)
            elseif direction == 2
                push!(escorts[nearest_id].itemsy, key)
            end
        end
    end
       
    
    return nearest_id
end
function path_to_io_exists_if(matrix, itemscoords, IO)
    if isa(IO, Vector)
        IO = isempty(itemscoords) ? IO[1] :
            argmin(io -> abs(io[1] - sum(c[1] for c in itemscoords)/length(itemscoords)) +
                         abs(io[2] - sum(c[2] for c in itemscoords)/length(itemscoords)), IO)
    end
    rows, cols = size(matrix)
    dir = IO[1]>size(matrix,1)/2 ? -1 : 1 # if io is on the left we have sink right, else on the left

    x_max= dir == 1 ? min(size(matrix,1),(IO[1] + length(itemscoords))) : max(1, IO[1] - length(itemscoords)); y_max =  IO[2]

    # Mark future positions
    future_blocked = zeros(Int, rows, cols)
    for pair in itemscoords
        x, y = pair
        if pair != IO
            future_blocked[x, y] = 1
        end
    end

    x_max = min(x_max, rows)
    y_max = min(y_max, cols)
    startcoords = (x_max, y_max)
    target_dist = abs(startcoords[1] - IO[1]) + abs(startcoords[2] - IO[2])

    # A* search setup
    visited = zeros(Int, rows, cols)
    dist = fill(Inf, rows, cols)
    open_set = BinaryMinHeap{Tuple{Float64, Int, Int}}()  # Store (priority, x, y)

    # Manhattan heuristic toward startcoords
    h(x, y) = abs(x - startcoords[1]) + abs(y - startcoords[2])
    dist[IO[1], IO[2]] = 0
    priority = dist[IO[1], IO[2]] + h(IO[1], IO[2])
    push!(open_set, (priority, IO[1], IO[2]))

    # A* loop
    while !isempty(open_set)
        (priority, cx, cy) = pop!(open_set)

        if visited[cx, cy] == 1
            continue
        end
        visited[cx, cy] = 1

        # succeed if any cell at the target Manhattan distance from IO is reached
        if abs(cx - IO[1]) + abs(cy - IO[2]) >= target_dist
            return true
        end

        # Explore neighbors
        for (nx, ny) in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]
            if 1 ≤ nx ≤ rows && 1 ≤ ny ≤ cols && future_blocked[nx, ny] != 1
                cost_here = dist[cx, cy] + 1
                if cost_here < dist[nx, ny]
                    dist[nx, ny] = cost_here
                    priority = cost_here + h(nx, ny)
                    push!(open_set, (priority, nx, ny))
                end
            end
        end
    end
    #println("No path found to start coordinates: ", startcoords)
    #print_matrix(future_blocked, visited)
    return false
end
function outwards_astar_with_dirchange(matrix, IO, blockmat, escortid, escorts, items; distval = 0.01)
    if isa(IO, Tuple)
        iox, ioy = IO
        rows, cols = size(matrix)
        allkeys = vcat(keys(escorts), keys(items))
        itemcoords = [(items[key].coords[1], items[key].coords[2]) for key in keys(items)]
        escortcoords = [(escorts[key].coords[1], escorts[key].coords[2]) for key in keys(escorts) if key != escortid]
        allcoords = itemcoords# vcat(itemcoords, escortcoords)
        x_max, y_max = escorts[escortid].coords
    
        # Mark future positions
        future_blocked = deepcopy(blockmat)
        for pair in allcoords
            x, y = pair
            if pair != IO
                future_blocked[x, y] = 1
            end
        end
     
    
        startcoords = (x_max, y_max)
    
        # Directions: 1 = none/initial, 2 = horizontal, 3 = vertical
        dist = fill(Inf, rows, cols, 3)
        visited = fill(false, rows, cols, 3)
    
        # If x1 == x2 => movement must be vertical (3), else horizontal (2)
        function dir_type(x1, y1, x2, y2)
            return (x1 == x2) ? 3 : 2
        end
    
        open_set = BinaryMinHeap{Tuple{Float64, Int, Int, Int}}()
    
        # Manhattan heuristic
        h(x, y) = 0.2* (abs(x - startcoords[1]) + abs(y - startcoords[2]))
    
        # Initialize at IO with direction = 1 (none/initial)
        dist[IO[1], IO[2], 1] = 0
        init_priority = dist[IO[1], IO[2], 1] + h(IO[1], IO[2])
        push!(open_set, (init_priority, IO[1], IO[2], 1))
    
        found_path = false
    
        # A* loop
        while !isempty(open_set)
            (priority, cx, cy, cdir) = pop!(open_set)
    
            if visited[cx, cy, cdir]
                continue
            end
            visited[cx, cy, cdir] = true
    
            for (nx, ny) in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]
                if 1 ≤ nx ≤ rows && 1 ≤ ny ≤ cols && future_blocked[nx, ny] != 1
                    ndir = dir_type(cx, cy, nx, ny)
                    # +1 for moving a step, plus +1 more if changing direction (excluding first step)
                    extra_cost = (cdir == 1 || cdir == ndir) ? 0 : 1  # Reduce penalty to 0.5
                    cost_here = dist[cx, cy, cdir] +  extra_cost + distval
                    if cost_here < dist[nx, ny, ndir]
                        dist[nx, ny, ndir] = cost_here
                        new_priority = cost_here + h(nx, ny)
                        push!(open_set, (new_priority, nx, ny, ndir))
                    end
                end
            end
            if (cx, cy) == startcoords
                # Ensure all slots between IOx and escx at coordinate escy are explored
                x_step = sign(cx - iox)  # Determine direction of iteration
                all_x_explored = true
                if x_step !=0
                    for x in iox:x_step:cx
                        if !visited[x, cy, cdir] && future_blocked[x, cy] != 1
                            ndir = dir_type(cx, cy, x, cy)
                            extra_cost = (cdir == 1 || cdir == ndir) ? 0 : 1
                            cost_here = dist[cx, cy, cdir] + extra_cost
                
                            if cost_here < dist[x, cy, ndir]
                                dist[x, cy, ndir] = cost_here
                                new_priority = cost_here + h(x, cy)
                                push!(open_set, (new_priority, x, cy, ndir))
                                visited[x, cy, cdir] = true  
                            end
                        end
                    end
                    all_x_explored = all(visited[x, cy, cdir] || future_blocked[x, cy] == 1 for x in iox:x_step:cx)
                end
    
            
                # Ensure all slots between IOy and escy at coordinate escx are explored
                y_step = sign(cy - ioy)  # Determine direction of iteration
                all_y_explored = true
                if y_step != 0 
                    for y in ioy:y_step:cy
                        if !visited[cx, y, cdir] && future_blocked[cx, y] != 1
                            ndir = dir_type(cx, cy, cx, y)
                            extra_cost = (cdir == 1 || cdir == ndir) ? 0 : 1
                            cost_here = dist[cx, cy, cdir] + extra_cost
                
                            if cost_here < dist[cx, y, ndir]
                                dist[cx, y, ndir] = cost_here
                                new_priority = cost_here + h(cx, y)
                                push!(open_set, (new_priority, cx, y, ndir))
                                visited[cx, y, cdir] = true  
                            end
                        end
                    end
                    all_y_explored = all(visited[cx, y, cdir] || future_blocked[cx, y] == 1 for y in ioy:y_step:cy)
                end
            
                
            
                if all_x_explored && all_y_explored  #  Only mark found if the full range is covered
                    found_path = true
                    break
                end
            end
        end
    
        # Convert dist into a 2D cost by taking the minimum cost ignoring direction
        min_cost = fill(Inf, rows, cols)
        for x in 1:rows
            for y in 1:cols
                if distval == 0.0
                    min_cost[x, y] = floor(minimum(dist[x, y, 1:3]))
                else
                    min_cost[x, y] = minimum(dist[x, y, 1:3])
                end
            end
        end
    
        return found_path, min_cost
    elseif isa(IO, Vector{Tuple{Int,Int}})
        combined_astar_matrices = zeros(Float64, rows, cols, num_io)
        for i in 1:num_io
            io = IO[i]
            worked, asternmat = outwards_astar_with_dirchange(matrix, IO, blockmat, escortid, escorts, items, distval=0.01)
            if worked
                combined_astar_matrices[:, :, i] = asternmat
            end
        end
        
        rows, cols, _ = size(combined_astar_matrices)
        min_cost = fill(Inf, rows, cols)

        for x in 1:rows
            for y in 1:cols
                # Gather non-zero values across all IO layers
                vals = [combined_astar_matrices[x, y, i] for i in 1:num_io if combined_astar_matrices[x, y, i] != 0.0]
                # Take the minimum if any non-zero values exist, otherwise leave 0
                if !isempty(vals)
                    min_cost[x, y] = minimum(vals)
                end
            end
        end

        return true, min_cost
    end
    

end
"""moves one escort to the final coordinates, modifying the incumbent matrix and the positions of items and escorts"""
# Reach-escort guard: switched on by the stall rollback in main.jl (after
# STALL_WINDOW steps without any target load moving, the run is rolled back to
# just before the last target-load move and replayed with the guard on). While
# on, move_escort! rejects any move that cuts an escort off from the IO, i.e.
# afterwards fewer escorts can be reached from the IO by a path that avoids
# target loads. Kept in task-local storage so parallel runs don't see each
# other's guard.
reach_guard_io() = get(task_local_storage(), :reach_guard_io, nothing)
set_reach_guard!(io) = (task_local_storage(:reach_guard_io, io); nothing)

# Forced freeroam: switched on by main.jl after STALL_WINDOW steps in which no
# escort was assigned as a mover and no target load moved. While on,
# moveescorts_flow! still tries direct serve, but skips urgent serve and sends
# every remaining idle escort through freeroam! (never freeroam_dumb!). It switches itself off as soon as
# the mover phase has an escort assigned again. last_n_movers() reports how many
# movers the latest moveescorts_flow! call had (task-local, like the guard).
force_freeroam() = get(task_local_storage(), :force_freeroam, false)
set_force_freeroam!(b) = (task_local_storage(:force_freeroam, b); nothing)
last_n_movers() = get(task_local_storage(), :last_n_movers, -1)

function n_escorts_reachable_from_io(matrix, items, escorts, IO)
    ios = IO isa Tuple ? [IO] : IO
    rows, cols = size(matrix)
    seen = falses(rows, cols)
    queue = Tuple{Int,Int}[]
    for io in ios
        haskey(items, matrix[io...]) && return length(escorts)   # a target load is on the IO, about to be picked
        seen[io...] = true
        push!(queue, io)
    end
    n = 0
    head = 1
    while head <= length(queue)
        x, y = queue[head]; head += 1
        haskey(escorts, matrix[x, y]) && (n += 1)
        for (nx, ny) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
            if 1 <= nx <= rows && 1 <= ny <= cols && !seen[nx, ny] && !haskey(items, matrix[nx, ny])
                seen[nx, ny] = true
                push!(queue, (nx, ny))
            end
        end
    end
    return n
end

function move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
    if UNIT_STEP[] && SINGLE_ITEM_RUN[] && !isempty(items) &&
       all(it -> it.assigned_io isa Tuple  ? it.coords == it.assigned_io :
                 it.assigned_io isa Vector ? it.coords in it.assigned_io : false,
           values(items))
        return 0   # lone item already delivered — stop burning moves
    end
    xgoal, ygoal = escort_finalcoords
    xcurr, ycurr = escorts[escortid].coords
    if UNIT_STEP[]
        if xgoal != xcurr
            xgoal = xcurr + sign(xgoal - xcurr)
        elseif ygoal != ycurr
            ygoal = ycurr + sign(ygoal - ycurr)
        end
    end
    dest_had_escort = 1 <= xgoal <= size(matrix, 1) && 1 <= ygoal <= size(matrix, 2) &&
                      haskey(escorts, matrix[xgoal, ygoal]) && matrix[xgoal, ygoal] != escortid
    direction = 0
    if xgoal == xcurr && ygoal == ycurr
        return 0
    elseif xgoal == xcurr
        if ygoal> ycurr
            direction = 2 # move escort up, block down
        elseif ygoal < ycurr
            direction = -2 # move escort down, block up
        end
    elseif ygoal == ycurr
        if xgoal > xcurr
            direction = 1 # move escort right , block to left
        elseif xgoal < xcurr
            direction = -1  # move escort left , block to right
        end
    end

    # Reach-escort guard: remember the shifted segment so the move can be undone.
    # Only moves that cut off an escort that is currently reachable are rejected.
    guard_io = reach_guard_io()
    guarded = guard_io !== nothing && direction != 0
    if guarded
        n_reach_before = n_escorts_reachable_from_io(matrix, items, escorts, guard_io)
        segment = abs(direction) == 1 ? [(x, ycurr) for x in min(xcurr, xgoal):max(xcurr, xgoal)] :
                                        [(xcurr, y) for y in min(ycurr, ygoal):max(ycurr, ygoal)]
        saved_cells = [(c, matrix[c...]) for c in segment]
    end

    #Depending on the direction, move the block, update the coordinates of the items and escorts if they exist in the block
    if direction == 1
        for x in xcurr+1:xgoal
            cand_id = matrix[x, ycurr]
            if haskey(items, cand_id)
                items[cand_id].coords = (x-1, ycurr) # update item's coordinates
                mark_contrib!(cand_id, escortid)
            elseif haskey(escorts, cand_id)
                escorts[cand_id].coords = (x-1, ycurr) # update escort's coordinates
            end
            matrix[x-1, ycurr] = cand_id # move block to left
            matrix[x, ycurr] = ""
        end
    elseif direction == -1
        for x in xcurr-1:-1:xgoal
            cand_id = matrix[x, ycurr]
            if haskey(items, cand_id)
                items[cand_id].coords = (x+1, ycurr) # update item's coordinates
                mark_contrib!(cand_id, escortid)
            elseif haskey(escorts, cand_id)
                escorts[cand_id].coords = (x+1, ycurr) # update escort's coordinates
            end
            matrix[x+1, ycurr] = cand_id # move block to right
            matrix[x, ycurr] = ""
        end
    elseif direction == 2
        for y in ycurr+1:ygoal
            cand_id = matrix[xcurr, y]
            if haskey(items, cand_id)
                items[cand_id].coords = (xcurr, y-1) # update item's coordinates
                mark_contrib!(cand_id, escortid)
            elseif haskey(escorts, cand_id)
                escorts[cand_id].coords = (xcurr, y-1) # update escort's coordinates
            end
            matrix[xcurr, y-1] = cand_id # move block down
            matrix[xcurr, y] = ""
        end
    elseif direction == -2
        for y in ycurr-1:-1:ygoal
            cand_id = matrix[xcurr, y]
            if haskey(items, cand_id)
                items[cand_id].coords = (xcurr, y+1) # update item's coordinates
                mark_contrib!(cand_id, escortid)
            elseif haskey(escorts, cand_id)
                escorts[cand_id].coords = (xcurr, y+1) # update escort's coordinates
            end
            matrix[xcurr, y+1] = cand_id # move block up
            matrix[xcurr, y] = ""
        end
    end
    matrix[xgoal, ygoal] = escortid # update matrix
    escorts[escortid].coords = (xgoal, ygoal) # update escort's coordinates
    if guarded && n_escorts_reachable_from_io(matrix, items, escorts, guard_io) < n_reach_before
        for (c, id) in saved_cells # undo: this move would cut an escort off from the IO
            matrix[c...] = id
            haskey(items, id) && (items[id].coords = c)
            haskey(escorts, id) && (escorts[id].coords = c)
        end
        # ban this escort from the target loads it tried to push for the next few
        # iterations, so the assignment picks something else instead of retrying
        pushed = [id for (_, id) in saved_cells if haskey(items, id)]
        for it in CURRENT_ITER[]+1:CURRENT_ITER[]+3, id in pushed
            push!(get!(escorts[escortid].banset, it, String[]), id)
        end
        return 0
    end
    #print_matrix(matrix)
    TOTAL_MOVES[] += 1
    (UNIT_STEP[] && dest_had_escort) || (ESCORT_RELOCS[] += 1)
    return 1
end


"""
moves all escorts, starting with the mover escorts
"""
function moveescorts!(iteration, matrix, items, escorts, moverescortids, blockmat, IO)
# MOVERS FIRST

    iox, ioy = IO
    #if iteration == 5
    #    println("here")
    #end
    checkpathformovers = false
    esccoords = [(escorts[key].coords[1], escorts[key].coords[2]) for key in moverescortids]
    closeescorts = findall([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) <= (length(keys(items))) for coord in esccoords])
    if !isempty(closeescorts)
        checkpathformovers = true
    end
    for escortid in moverescortids
        FREEZE_ENABLED[] && !escort_active(escortid, iteration) && continue
        itemsx = escorts[escortid].itemsx
        itemsy = escorts[escortid].itemsy
        if !isempty(itemsx)
            direction = 1
            itemid = itemsx[1]
        elseif !isempty(itemsy)
            direction = 2
            itemid = itemsy[1]
        else
            println("No item found, check item escort assignment")
            continue
        end
        if itemid in Iterators.flatten(values(escorts[escortid].banset)) || 
            futurecoords_closetoIO(items,  itemid, escorts, escortid, direction, IO) 
            checkpathformovers = true
        end
        item = items[itemid]
        if (item.direction != direction) 
            println("Item direction and escort direction do not match")
        end

        itemx, itemy = item.coords
        escortx, escorty = escorts[escortid].coords
        # Here onwards until the move_escort! function, we find the nearest item where this escort could be useful in next time step
        candid,candx,candy = find_nearest_item_toitem(matrix, items, itemid, blockmat, IO, direction)
        gapx, gapy = abs(candx - iox), abs(candy - ioy)
        if direction == 1
            if candid == "" || candx == itemx
                escort_finalcoords = (itemx, escorty)
            else 
                
                if items[candid].direction == 1 ||  gapx < gapy # moving in x
                    if iox > min(itemx, candx) 
                        escort_finalcoords = (max(1, candx+1), itemy) # IO on the right
                    else iox < min(itemx, candx)
                        escort_finalcoords = (max(1, candx-1), itemy) # IO on the left
                    end
                elseif items[candid].direction == 2 || gapy <= gapx # moving in y
                    escort_finalcoords = (candx, itemy)
                end
            end
        elseif direction == 2
            if candid == "" || candy == itemy
                escort_finalcoords = (escortx, itemy)
            else 
                if items[candid].direction == 1 ||  gapx < gapy# moving in x
                    escort_finalcoords = (itemx, candy) 
                elseif items[candid].direction == 2 || gapy <= gapx# moving in y
                    escort_finalcoords = (itemx, max(1, candy-1))
                end
            end
        end
        if escort_finalcoords != ( escortx, escorty) # TODO add path tho io exists if in range
            if checkpathformovers
                itemscoords = generatefuturecoords_fincoord(items,  escorts, direction, escortid, escort_finalcoords, matrix, IO) 
                samecoords = direction == 2 ? 
                filter(x -> x[1] == itemx && x[2] >= min(itemy, escorty) && x[2] <= max(itemy, escorty), itemscoords) :
                filter(x -> x[2] == itemy && x[1] >= min(itemx, escortx) && x[1] <= max(itemx, escortx), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in samecoords])
                if minDist  > length(keys(items)) || # item far out from IO
                    path_to_io_exists_if(matrix, itemscoords, IO)   # check with A* if this movement would cause some stupid block
                    
                    push!(escorts[escortid].tabu, (escortx,escorty))
                    moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                    updateblockmat_e!(blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2])
                    escorts[escortid].lastmoved = iteration
    
                else# else we ban it for next iteration to simplify computation on assignment! 
                    if !haskey(escorts[escortid].banset, iteration+1)
                        escorts[escortid].banset[iteration+1] = [itemid]
                    else
                        push!(escorts[escortid].banset[iteration+1],itemid)
                    end
                    filter!(x -> x != escortid, moverescortids)
                    updateblockmat_e!(blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2], val=0) # unblock the path 

                end
            else
                push!(escorts[escortid].tabu, (escortx,escorty))
                moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                updateblockmat_e!(blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2])
                escorts[escortid].lastmoved = iteration
            end

           
        end
    end
    
    # First, filter the customers:
    urgentcustomers = filter(customer_id ->  floor(Int,iteration+ (abs(iox -items[customer_id].coords[1]) + items[customer_id].coords[2]) * 1.5) >= items[customer_id].deadline,keys(items))
    
    
    
    # URGENCY POLICY UNDER CONSTRUCTION, will need to go into the find nearest item to escort function i guess due to complexity
    
   
    
    # NON MOVERS (nonassigned in earlier stage)
    nonmovers = setdiff(keys(escorts), moverescortids)
    FREEZE_ENABLED[] && (nonmovers = Set(filter(e -> escort_active(e, iteration), collect(nonmovers))))
    # Sort non-movers according to the number of empty spaces in front of them in the y direction
    nonmovers = sort(collect(nonmovers), by = escortid -> begin
                esc_x, esc_y = escorts[escortid].coords
                empty_spaces = 0
                for y in esc_y-1:-1:1
                    if blockmat[esc_x, y] == 1 || haskey(items, matrix[esc_x, y]) || haskey(escorts, matrix[esc_x, y])
                        break
                    end
                    empty_spaces += 1
                end
                distance_to_IO = -euclidean_distance((esc_x, esc_y), IO)  # negative for descending
                    return (distance_to_IO, empty_spaces)
                end, rev=false) 

    usedescorts = String[]
    #Direct serve
    for escortid in nonmovers
        esc_x , esc_y = escorts[escortid].coords
        if blockmat[esc_x, esc_y] == 1
            continue
        end
        moved, escort_finalcoords = directserve_makespan!(iteration, matrix, items, escorts, escortid, urgentcustomers, blockmat, IO)
        if moved && escort_finalcoords != (esc_x,esc_y)
            push!(escorts[escortid].tabu, (esc_x,esc_y))
            push!(usedescorts,escortid)
            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
            updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
            escorts[escortid].lastmoved = iteration
        end
    end
    # Remove used escorts from nonmovers
    nonmovers = setdiff(nonmovers, usedescorts)
    #3-4 step serve
    usedescorts = String[]
    if !isempty(urgentcustomers)
        urgentmatrixes = urgmats(items, escorts, blockmat, matrix, urgentcustomers, IO)
        for escortid in nonmovers
            esc_x , esc_y = escorts[escortid].coords
            if blockmat[esc_x, esc_y] == 1
                continue
            end
            moved, escort_finalcoords = urgserve!(iteration, matrix, items, escorts, escortid, urgentmatrixes, IO)
            if moved && escort_finalcoords != (esc_x,esc_y)
                push!(escorts[escortid].tabu, (esc_x,esc_y))
                push!(usedescorts,escortid)
                moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                updateurgmats_e!(urgentmatrixes, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                escorts[escortid].lastmoved = iteration
            end
        end
    end
    nonmovers = setdiff(nonmovers, usedescorts)
    for escortid in nonmovers
        esc_x , esc_y = escorts[escortid].coords
        if blockmat[esc_x, esc_y] == 1
            continue
        end
        moved, escort_finalcoords = freeroam!(iteration, matrix, items, escorts, escortid, blockmat, IO)
        if moved && escort_finalcoords != (esc_x,esc_y)
            push!(escorts[escortid].tabu, (esc_x,esc_y))
            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
            updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
            escorts[escortid].lastmoved = iteration
        end
    end
    #checksync(matrix, escorts, items)
    #print_matrix(matrix, blockmat)
    return (moved_any>0)
end
"""
How far escortid can travel along the X axis from (esc_x, esc_y) toward
target_x before hitting a blocked cell, another escort, or an item — a
no-push repositioning move (occupied cells stop it, they are never shoved
through), same rule freeroam! already follows. Capped at target_x so the
escort never overshoots the alignment it's after.
"""
function clamp_reach_x(matrix, blockmat, escorts, items, escortid, esc_x, esc_y, target_x)
    dir = target_x > esc_x ? 1 : -1
    reach = esc_x
    x = esc_x
    while x != target_x
        x += dir
        blockmat[x, esc_y] == 1 && break
        cell = matrix[x, esc_y]
        ((haskey(escorts, cell) && cell != escortid) || haskey(items, cell)) && break
        reach = x
    end
    return reach
end

"""Y-axis counterpart of clamp_reach_x — see its docstring."""
function clamp_reach_y(matrix, blockmat, escorts, items, escortid, esc_x, esc_y, target_y)
    dir = target_y > esc_y ? 1 : -1
    reach = esc_y
    y = esc_y
    while y != target_y
        y += dir
        blockmat[esc_x, y] == 1 && break
        cell = matrix[esc_x, y]
        ((haskey(escorts, cell) && cell != escortid) || haskey(items, cell)) && break
        reach = y
    end
    return reach
end

"""
LM mode only. After an escort pushes an item, it lands adjacent to the item
on the axis it just served (the "base escort", Yalcin et al. 2019) — free to
reuse if the item continues in the same direction, but not aligned for the
other axis if the item needs to switch. Left to the generic freeroam!/
directserve! tail, that escort has no idea it's needed again and can wander
off (confirmed visually: an escort one step from being useful would drift
away for 1-2 iterations before repositioning back).

This walks each item's last-serving escort (COMMITTED_ESCORT) directly
toward alignment on whichever axis it currently lacks, closing the smaller
gap first — the same one-move-reaches-alignment trick as the assignment
stage's sticky-direction logic — instead of leaving it to generic
repositioning. Only intervenes when the item is genuinely stalled
(direction == 0, i.e. assignment found no candidate at all this iteration)
and the escort is idle; a repositioned escort is removed from `nonmovers` so
freeroam!/directserve! don't also try to place it.
"""
function reposition_base_escorts!(iteration, matrix, items, escorts, nonmovers, blockmat)
    used = String[]
    moved_any_local = 0
    for (itemid, escid) in COMMITTED_ESCORT
        haskey(items, itemid) || continue
        item = items[itemid]
        item.direction == 0 || continue
        (haskey(escorts, escid) && escid in nonmovers) || continue

        esc_x, esc_y = escorts[escid].coords
        blockmat[esc_x, esc_y] == 1 && continue
        itemx, itemy = item.coords
        (esc_y == itemy || esc_x == itemx) && continue  # already aligned on some axis

        dx, dy = itemx - esc_x, itemy - esc_y
        finalcoords = if abs(dx) <= abs(dy)
            (clamp_reach_x(matrix, blockmat, escorts, items, escid, esc_x, esc_y, itemx), esc_y)
        else
            (esc_x, clamp_reach_y(matrix, blockmat, escorts, items, escid, esc_x, esc_y, itemy))
        end

        if finalcoords != (esc_x, esc_y)
            push!(escorts[escid].tabu, (esc_x, esc_y))
            push!(used, escid)
            moved_any_local += move_escort!(matrix, items, escorts, escid, finalcoords)
            updateblockmat_e!(blockmat, esc_x, esc_y, finalcoords[1], finalcoords[2])
            escorts[escid].lastmoved = iteration
        end
    end
    return used, moved_any_local
end

function moveescorts_flow!(iteration, matrix, items, escorts, moverescortids, blockmat, IO)
    # MOVERS FIRST
    CURRENT_ITER[] = iteration

    iox, ioy = IO
    moved_any = 0
    checkpathformovers = false
    task_local_storage(:last_n_movers, length(moverescortids))
    !isempty(moverescortids) && set_force_freeroam!(false)   # an escort is aligned again: stop forcing
    forced_freeroam = force_freeroam()

    # Process movers starting with the escort closest to its own target item
    # (whichever of itemsx/itemsy is populated), ascending. Escorts with no
    # tracked item (shouldn't normally happen) sort last via Inf.
    function escort_item_distance(escortid)
        ex, ey = escorts[escortid].coords
        itemid = !isempty(escorts[escortid].itemsx) ? escorts[escortid].itemsx[1] :
                 !isempty(escorts[escortid].itemsy) ? escorts[escortid].itemsy[1] : ""
        itemid == "" && return Inf
        ix, iy = items[itemid].coords
        return abs(ex - ix) + abs(ey - iy)
    end
    moverescortids = sort(moverescortids, by = escort_item_distance)

    esccoords = [(escorts[key].coords[1], escorts[key].coords[2]) for key in moverescortids]
    closeescorts = findall([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) <= (length(keys(items))) for coord in esccoords])
    if !isempty(closeescorts)
        checkpathformovers = true
    end
    DEBUG_MOVE_TRACE[] && println("  MOVERS at t=$iteration: $moverescortids  item coords: $(Dict(k=>v.coords for (k,v) in items))")
    if iteration == 2 && IO == (1,1) && haskey(escorts, "E1") && haskey(escorts, "E2") && haskey(items, "I1") &&
       escorts["E1"].coords == (3,3) && escorts["E2"].coords == (1,2) && items["I1"].coords == (4,2)
        println("INST20_T2: escorts=$(Dict(k=>v.coords for (k,v) in escorts)) items=$(Dict(k=>v.coords for (k,v) in items)) movers=$moverescortids blockmat_col3=$(blockmat[3,:])")
    end
    for escortid in moverescortids
        FREEZE_ENABLED[] && !escort_active(escortid, iteration) && continue
        itemsx = escorts[escortid].itemsx
        itemsy = escorts[escortid].itemsy
        if !isempty(itemsx)
            direction = 1
            itemid = itemsx[1]
        elseif !isempty(itemsy)
            direction = 2
            itemid = itemsy[1]
        else
            println("No item found, check item escort assignment")
            continue
        end
        if itemid in Iterators.flatten(values(escorts[escortid].banset)) || 
            futurecoords_closetoIO(items,  itemid, escorts, escortid, direction, IO) 
            checkpathformovers = true
        end
        item = items[itemid]
        if (item.direction != direction) 
            println("Item direction and escort direction do not match")
        end

        itemx, itemy = item.coords
        escortx, escorty = escorts[escortid].coords
        # Here onwards until the move_escort! function, we find the nearest item where this escort could be useful in next time step
        candid,candx,candy = find_nearest_item_toitem(matrix, items, itemid, blockmat, IO, direction)
        gapx, gapy = abs(candx - iox), abs(candy - ioy)
        DEBUG_MOVE_TRACE[] && println("  [MOVER] t=$iteration esc=$escortid from=($escortx,$escorty) dir=$direction item=$itemid itempos=($itemx,$itemy) cand=$candid candpos=($candx,$candy)")
        if direction == 1
            if candid == "" || candx == itemx
                escort_finalcoords = (itemx, escorty)
            else 
                
                if items[candid].direction == 1 ||  gapx < gapy # moving in x
                    if iox > min(itemx, candx) 
                        escort_finalcoords = (max(1, candx+1), itemy) # IO on the right
                    else iox < min(itemx, candx)
                        escort_finalcoords = (max(1, candx-1), itemy) # IO on the left
                    end
                elseif items[candid].direction == 2 || gapy <= gapx # moving in y
                    escort_finalcoords = (candx, itemy)
                end
            end
        elseif direction == 2
            if candid == "" || candy == itemy
                escort_finalcoords = (escortx, itemy)
            else 
                if items[candid].direction == 1 ||  gapx < gapy# moving in x
                    escort_finalcoords = (itemx, candy) 
                elseif items[candid].direction == 2 || gapy <= gapx# moving in y
                    escort_finalcoords = (itemx, max(1, candy-1))
                end
            end
        end
        if escort_finalcoords != ( escortx, escorty) # TODO add path tho io exists if in range
            # This risk check judges the uncapped, potentially multi-cell
            # escort_finalcoords target -- but under LM mode move_escort! only
            # ever actually executes a single-cell step toward it, so the
            # check would ban safe 1-cell moves based on a jump that was
            # never going to happen. Bypass it in LM mode.
            if checkpathformovers && (!UNIT_STEP[] || LM_PATHCHECK[])   # path-to-IO check: BM always, LM when LM_PATHCHECK[]
                itemscoords = generatefuturecoords_fincoord(items,  escorts, direction, escortid, escort_finalcoords, matrix, IO) 
                samecoords = direction == 2 ? 
                filter(x -> x[1] == itemx && x[2] >= min(itemy, escorty) && x[2] <= max(itemy, escorty), itemscoords) :
                filter(x -> x[2] == itemy && x[1] >= min(itemx, escortx) && x[1] <= max(itemx, escortx), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in samecoords])
                if minDist  > length(keys(items)) || # item far out from IO
                    path_to_io_exists_if(matrix, itemscoords, IO)   # check with A* if this movement would cause some stupid block
                    
                    push!(escorts[escortid].tabu, (escortx,escorty))
                    moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                    updateblockmat_e!(blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2])
                    escorts[escortid].lastmoved = iteration
    
                else# else we ban it for next iteration to simplify computation on assignment! 
                    if !haskey(escorts[escortid].banset, iteration+1)
                        escorts[escortid].banset[iteration+1] = [itemid]
                    else
                        push!(escorts[escortid].banset[iteration+1],itemid)
                    end
                    filter!(x -> x != escortid, moverescortids)
                    updateblockmat_e!(blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2], val=0) # unblock the path 

                end
            else
                push!(escorts[escortid].tabu, (escortx,escorty))
                moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                # move_escort! clamps to a single cell under LM mode, so the
                # escort's actual resting place can be short of escort_finalcoords
                # (the uncapped intended target). Marking blockmat with the
                # uncapped target leaves a phantom blocked corridor over cells
                # the escort never actually reached, which then makes OTHER
                # escorts' path checks (e.g. directserve_flow!'s path_blocked)
                # see a false blockage. Mark only the cells actually traversed.
                actual_x, actual_y = escorts[escortid].coords
                updateblockmat_e!(blockmat, escortx, escorty, actual_x, actual_y)
                escorts[escortid].lastmoved = iteration
                DEBUG_MOVE_TRACE[] && println("  [MOVER-move] t=$iteration esc=$escortid from=($escortx,$escorty) target=$escort_finalcoords actual=($actual_x,$actual_y)")
            end

           
        end
    end
        
    diagonal_size = sqrt(size(matrix, 1)^2 + size(matrix, 2)^2)
    # First, filter the customers:
    urgentcustomers = filter(customer_id -> begin
        item = items[customer_id]
        floor(Int, iteration + (abs(iox - item.coords[1]) + item.coords[2]) * 1.5) >= item.deadline ||
        (iteration - item.tes) + (abs(item.coords[1] - iox) + abs(item.coords[2] - ioy)) > diagonal_size
    end, keys(items))
    
    
    # URGENCY POLICY UNDER CONSTRUCTION, will need to go into the find nearest item to escort function i guess due to complexity
    
    
    
    # NON MOVERS (nonassigned in earlier stage)
    nonmovers = setdiff(keys(escorts), moverescortids)
    FREEZE_ENABLED[] && (nonmovers = Set(filter(e -> escort_active(e, iteration), collect(nonmovers))))
    # Sort non-movers according to the number of empty spaces in front of them in the y direction
    nonmovers = sort(collect(nonmovers), by = escortid -> begin
            esc_x, esc_y = escorts[escortid].coords
            empty_spaces = 0
            for y in esc_y-1:-1:1
                if blockmat[esc_x, y] == 1 || haskey(items, matrix[esc_x, y]) || haskey(escorts, matrix[esc_x, y])
                    break
                end
                empty_spaces += 1
            end
            distance_to_IO = -euclidean_distance((esc_x, esc_y), IO)  # negative for descending
                return (distance_to_IO, empty_spaces)
            end, rev=false)

    if UNIT_STEP[] && BASE_ESCORT_REPOSITION[]
        base_used, base_moved = reposition_base_escorts!(iteration, matrix, items, escorts, nonmovers, blockmat)
        moved_any += base_moved
        nonmovers = setdiff(nonmovers, base_used)
    end

    usedescorts = String[]
    #Direct serve
    for escortid in nonmovers
        esc_x , esc_y = escorts[escortid].coords
        if blockmat[esc_x, esc_y] == 1
            continue
        end
        moved, escort_finalcoords = directserve_flow!(iteration, matrix, items, escorts, escortid, urgentcustomers, blockmat, IO)
        if moved
            # UNIT_STEP[] mode allows directserve_flow! to return a same-coords
            # "stay here, I'm already aligned to serve next iteration" signal
            # (escort_finalcoords == (esc_x,esc_y)). Treat that as handled too
            # -- mark it used so urgserve!/freeroam! don't move it off its spot
            # -- without calling move_escort! (there's nothing to move).
            push!(usedescorts,escortid)
            if escort_finalcoords != (esc_x,esc_y)
                push!(escorts[escortid].tabu, (esc_x,esc_y))
                moved_any +=move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                escorts[escortid].lastmoved = iteration
            end
        end
        DEBUG_MOVE_TRACE[] && println("  [directserve_flow!] t=$iteration esc=$escortid from=($esc_x,$esc_y) moved=$moved target=$escort_finalcoords actual=$(escorts[escortid].coords)")
    end
    # Remove used escorts from nonmovers
    nonmovers = setdiff(nonmovers, usedescorts)
    #3-4 step serve
    usedescorts = String[]
    if !isempty(urgentcustomers) && !forced_freeroam   # forced freeroam: skip urgent serve (direct serve still runs)
        urgentmatrixes = urgmats(items, escorts, blockmat, matrix, urgentcustomers, IO)
        for escortid in nonmovers
            esc_x , esc_y = escorts[escortid].coords
            if blockmat[esc_x, esc_y] == 1
                continue
            end
            moved, escort_finalcoords = urgserve!(iteration, matrix, items, escorts, escortid, urgentmatrixes, IO)
            if moved && escort_finalcoords != (esc_x,esc_y)
                push!(escorts[escortid].tabu, (esc_x,esc_y))
                push!(usedescorts,escortid)
                moved_any +=move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                updateurgmats_e!(urgentmatrixes, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                escorts[escortid].lastmoved = iteration
            end
            DEBUG_MOVE_TRACE[] && println("  [urgserve!] t=$iteration esc=$escortid from=($esc_x,$esc_y) moved=$moved target=$escort_finalcoords actual=$(escorts[escortid].coords)")
        end
    end
    nonmovers = setdiff(nonmovers, usedescorts)

      smart = zeros(Int, length(nonmovers))
    if !UNIT_STEP[]  # LM mode: always use freeroam! (never freeroam_dumb!) — see below
        if length(nonmovers) > length(keys(items))
            gap = length(nonmovers) - length(keys(items))
            for i in 1:gap
                smart[i] = smart[end - gap + i] = 1
            end
        else
            if !isempty(smart)
                smart[end] = 1
            end
        end
    end

    for (index,escortid) in enumerate(nonmovers)
        esc_x , esc_y = escorts[escortid].coords
        moved= false
        if blockmat[esc_x, esc_y] == 1
            continue
        end
        traced_fn = ""
        tabu_before = copy(escorts[escortid].tabu)
        if smart[index] == 1 && !forced_freeroam
            moved, escort_finalcoords = freeroam_dumb!(iteration, matrix, items, escorts, escortid, blockmat, IO)
            traced_fn = "freeroam_dumb!"
        else
            moved, escort_finalcoords = freeroam!(iteration, matrix, items, escorts, escortid, blockmat, IO)
            traced_fn = "freeroam!"
        end
        if moved && escort_finalcoords != (esc_x,esc_y)
            # freeroam!/freeroam_dumb! check their (possibly multi-cell, BM-style)
            # candidate target against tabu, but under LM mode move_escort! then
            # clamps that target down to a single step -- the actual landing
            # cell was never itself checked, so an escort can get sent right
            # back onto the cell it just tabu'd itself off of (e.g. freeroam!
            # proposes (4,3), which isn't tabu, but the LM-clamped step lands on
            # tabu'd (4,2) anyway). Recompute the real landing cell the same way
            # move_escort! does and reject the move if that specific cell is tabu.
            landing = escort_finalcoords
            if UNIT_STEP[]
                lx, ly = escort_finalcoords
                if lx != esc_x
                    landing = (esc_x + sign(lx - esc_x), esc_y)
                elseif ly != esc_y
                    landing = (esc_x, esc_y + sign(ly - esc_y))
                end
            end
            if !(UNIT_STEP[] && landing in escorts[escortid].tabu)
                push!(escorts[escortid].tabu, (esc_x,esc_y))
                moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                escorts[escortid].lastmoved = iteration
            else
                moved = false
            end
        end
        DEBUG_MOVE_TRACE[] && println("  [$traced_fn] t=$iteration esc=$escortid from=($esc_x,$esc_y) moved=$moved target=$escort_finalcoords actual=$(escorts[escortid].coords) tabu_before=$tabu_before")
    end

    #checksync(matrix, escorts, items)
    print_matrix(matrix, blockmat)
    return (moved_any>0)
    #return matrix
end
function moveescorts_flow_r!(iteration, matrix, items, escorts, moverescortids, blockmat, IO)
    # MOVERS FIRST
    iox, ioy = IO
    moved_any = 0
    checkpathformovers = false

    # GRASP version of moveescorts_flow!'s nearest-escort-to-its-item mover
    # ordering (see sort_movers_by_item_distance_grasp docstring).
    moverescortids = sort_movers_by_item_distance_grasp(moverescortids, escorts, items, 0.8)

    esccoords = [(escorts[key].coords[1], escorts[key].coords[2]) for key in moverescortids]
    closeescorts = findall([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) <= (length(keys(items))) for coord in esccoords])
    if !isempty(closeescorts)
        checkpathformovers = true
    end
    for escortid in copy(moverescortids)
        itemsx = escorts[escortid].itemsx
        itemsy = escorts[escortid].itemsy
        if !isempty(itemsx) && !isempty(itemsy)
            # Both directions available: randomize which to serve and which item within it.
            if rand(Random.default_rng(), Bool)
                direction = 1; itemid = rand(Random.default_rng(), itemsx)
            else
                direction = 2; itemid = rand(Random.default_rng(), itemsy)
            end
        elseif !isempty(itemsx)
            direction = 1; itemid = rand(Random.default_rng(), itemsx)
        elseif !isempty(itemsy)
            direction = 2; itemid = rand(Random.default_rng(), itemsy)
        else
            #println("No item found (flow_r!): escort $escortid  itemsx=$(escorts[escortid].itemsx)  itemsy=$(escorts[escortid].itemsy)  all_moverescortids=$moverescortids")
            continue
        end
        if itemid in Iterators.flatten(values(escorts[escortid].banset)) ||
            futurecoords_closetoIO(items,  itemid, escorts, escortid, direction, IO)
            checkpathformovers = true
        end
        item = items[itemid]
        if (item.direction != direction) 
            println("Item direction and escort direction do not match")
        end

        itemx, itemy = item.coords
        escortx, escorty = escorts[escortid].coords
        # Here onwards until the move_escort! function, we find the nearest item where this escort could be useful in next time step
        candid,candx,candy = find_nearest_item_toitem(matrix, items, itemid, blockmat, IO, direction)
        gapx, gapy = abs(candx - iox), abs(candy - ioy)
        if direction == 1
            if candid == "" || candx == itemx
                escort_finalcoords = (itemx, escorty)
            else 
                
                if items[candid].direction == 1 ||  gapx < gapy # moving in x
                    if iox > min(itemx, candx) 
                        escort_finalcoords = (max(1, candx+1), itemy) # IO on the right
                    else iox < min(itemx, candx)
                        escort_finalcoords = (max(1, candx-1), itemy) # IO on the left
                    end
                elseif items[candid].direction == 2 || gapy <= gapx # moving in y
                    escort_finalcoords = (candx, itemy)
                end
            end
        elseif direction == 2
            if candid == "" || candy == itemy
                escort_finalcoords = (escortx, itemy)
            else 
                if items[candid].direction == 1 ||  gapx < gapy# moving in x
                    escort_finalcoords = (itemx, candy) 
                elseif items[candid].direction == 2 || gapy <= gapx# moving in y
                    escort_finalcoords = (itemx, max(1, candy-1))
                end
            end
        end
        if escort_finalcoords != ( escortx, escorty) # TODO add path tho io exists if in range
            if checkpathformovers
                itemscoords = generatefuturecoords_fincoord(items,  escorts, direction, escortid, escort_finalcoords, matrix, IO) 
                samecoords = direction == 2 ? 
                filter(x -> x[1] == itemx && x[2] >= min(itemy, escorty) && x[2] <= max(itemy, escorty), itemscoords) :
                filter(x -> x[2] == itemy && x[1] >= min(itemx, escortx) && x[1] <= max(itemx, escortx), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in samecoords])
                if minDist  > length(keys(items)) || # item far out from IO
                    path_to_io_exists_if(matrix, itemscoords, IO)   # check with A* if this movement would cause some stupid block
                    
                    push!(escorts[escortid].tabu, (escortx,escorty))
                    moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                    updateblockmat_e!(blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2])
                    escorts[escortid].lastmoved = iteration
    
                else# else we ban it for next iteration to simplify computation on assignment! 
                    if !haskey(escorts[escortid].banset, iteration+1)
                        escorts[escortid].banset[iteration+1] = [itemid]
                    else
                        push!(escorts[escortid].banset[iteration+1],itemid)
                    end
                    filter!(x -> x != escortid, moverescortids)
                    updateblockmat_e!(blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2], val=0) # unblock the path 

                end
            else
                push!(escorts[escortid].tabu, (escortx,escorty))
                moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                updateblockmat_e!(blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2])
                escorts[escortid].lastmoved = iteration
            end

           
        end
    end
        
    diagonal_size = sqrt(size(matrix, 1)^2 + size(matrix, 2)^2)
    # First, filter the customers:
    urgentcustomers = filter(customer_id -> begin
        item = items[customer_id]
        floor(Int, iteration + (abs(iox - item.coords[1]) + item.coords[2]) * 1.5) >= item.deadline ||
        (iteration - item.tes) + (abs(item.coords[1] - iox) + abs(item.coords[2] - ioy)) > diagonal_size
    end, keys(items))
    
    
    # URGENCY POLICY UNDER CONSTRUCTION, will need to go into the find nearest item to escort function i guess due to complexity
    
    
    
    # NON MOVERS (nonassigned in earlier stage)
    nonmovers = setdiff(keys(escorts), moverescortids)
    FREEZE_ENABLED[] && (nonmovers = Set(filter(e -> escort_active(e, iteration), collect(nonmovers))))
    # RANDOMIZATION: use GRASP sort with α=0.8 instead of the deterministic sort.
    # α=0.8 means escorts within 80% of the distance range are RCL candidates,
    # then one is picked uniformly at random — only the closest escorts to IO are excluded.
    nonmovers = sort_nonmovers_grasp(nonmovers, escorts, items, matrix, blockmat, IO, 0.3)

    usedescorts = String[]
    #Direct serve
    for escortid in nonmovers
        esc_x , esc_y = escorts[escortid].coords
        if blockmat[esc_x, esc_y] == 1
            continue
        end
        moved, escort_finalcoords = directserve_flow_r!(iteration, matrix, items, escorts, escortid, urgentcustomers, blockmat, IO)
        if moved && escort_finalcoords != (esc_x,esc_y)
            push!(escorts[escortid].tabu, (esc_x,esc_y))
            push!(usedescorts,escortid)
            moved_any +=move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
            updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
            escorts[escortid].lastmoved = iteration
        end
    end
    # Remove used escorts from nonmovers
    nonmovers = setdiff(nonmovers, usedescorts)
    #3-4 step serve
    usedescorts = String[]
    if !isempty(urgentcustomers)
        urgentmatrixes = urgmats(items, escorts, blockmat, matrix, urgentcustomers, IO)
        for escortid in nonmovers
            esc_x , esc_y = escorts[escortid].coords
            if blockmat[esc_x, esc_y] == 1
                continue
            end
            moved, escort_finalcoords = urgserve_r!(iteration, matrix, items, escorts, escortid, urgentmatrixes, IO)
            if moved && escort_finalcoords != (esc_x,esc_y)
                push!(escorts[escortid].tabu, (esc_x,esc_y))
                push!(usedescorts,escortid)
                moved_any +=move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                updateurgmats_e!(urgentmatrixes, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                escorts[escortid].lastmoved = iteration
            end
        end
    end
    #= nonmovers = setdiff(nonmovers, usedescorts)
    smart = zeros(Int, length(nonmovers))
    if length(nonmovers) > length(keys(items))
        gap = length(nonmovers) - length(keys(items))
        for i in 1:gap
            smart[i] = smart[end - gap + i] = 1
        end
    else
        if !isempty(smart)
            smart[end] = 1
        end
    end

    for (index,escortid) in enumerate(nonmovers)
        esc_x , esc_y = escorts[escortid].coords
        if blockmat[esc_x, esc_y] == 1
            continue
        end
        if smart[index] == 1
            moved, escort_finalcoords = freeroam_dumb!(iteration, matrix, items, escorts, escortid, blockmat, IO)
        else
            moved, escort_finalcoords = freeroam!(iteration, matrix, items, escorts, escortid, blockmat, IO)
        end
        if moved && escort_finalcoords != (esc_x,esc_y)
            push!(escorts[escortid].tabu, (esc_x,esc_y))
            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
            updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
            escorts[escortid].lastmoved = iteration
        end
    end
 =#

   
    nonmovers = setdiff(nonmovers, usedescorts)
#=       smart = zeros(Int, length(nonmovers))
    if length(nonmovers) > length(keys(items))
        gap = length(nonmovers) - length(keys(items))
        for i in 1:gap
            smart[i] = smart[end - gap + i] = 1
        end
    else
        if !isempty(smart)
            smart[end] = 1
        end
    end
    
    for (index,escortid) in enumerate(nonmovers)
        esc_x , esc_y = escorts[escortid].coords
        moved= false
        if blockmat[esc_x, esc_y] == 1
            continue
        end
        if smart[index] == 1
            moved, escort_finalcoords = freeroam_dumb!(iteration, matrix, items, escorts, escortid, blockmat, IO)
        end
        if moved && escort_finalcoords != (esc_x,esc_y)
            push!(escorts[escortid].tabu, (esc_x,esc_y))
            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
            updateblockmat_e!(blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
            escorts[escortid].lastmoved = iteration
        end
    end 
    smartesc = [escortid for (index, escortid) in enumerate(nonmovers) if smart[index] == 0] =#
    nonmovers_free = [eid for eid in nonmovers
                      if blockmat[escorts[eid].coords[1], escorts[eid].coords[2]] != 1]
    moved_any += cooperative_freeroam!(iteration, matrix, items, escorts, nonmovers_free, blockmat, IO)


    #checksync(matrix, escorts, items)
    return (moved_any>0)
end
# Under LM an escort steps just one cell toward escort_finalcoords (x-axis first,
# then y — see move_escort!). directserve/urgserve only tabu-check the far,
# uncapped target, so a two-escort ping-pong (each stepping one cell toward
# alternating IOs every iteration) never trips tabu. Predict the realized cell so
# the caller can reject a step that lands back on a just-vacated square.
function _lm_realized_step(cur, goal)
    cx, cy = cur; gx, gy = goal
    return gx != cx ? (cx + sign(gx - cx), cy) :
           gy != cy ? (cx, cy + sign(gy - cy)) : (cx, cy)
end

"""
Multi-IO version of moveescorts_flow!
Uses global_escort_items structure instead of individual escort.itemsx/itemsy
Processes mover escorts using IO-specific information and blockmats
"""
function moveescorts_flow_multi_io!(iteration, matrix, items, escorts, global_blockmat,
                                     global_escort_items, all_ios, item_to_ios)
    # MOVERS FIRST - Process using global_escort_items structure
    # global_escort_items[escort_id] = [(io, itemsx, itemsy), ...]
    CURRENT_ITER[] = iteration
  #=   if (iteration ==6)
        println("here")
    end =#
    moverescortids = collect(keys(global_escort_items))
    moved_any = 0
    checkpathformovers = false

    if TRACE_ESC[] != "" && haskey(escorts, TRACE_ESC[])
        _te = TRACE_ESC[]
        println("t=$iteration  $_te at $(escorts[_te].coords)  mover=$(_te in moverescortids)  tabu=$(escorts[_te].tabu)")
    end

    for escortid in moverescortids
        if !haskey(global_escort_items, escortid)
            continue
        end
        FREEZE_ENABLED[] && !escort_active(escortid, iteration) && continue
        
        assignments = global_escort_items[escortid]  # List of (io, itemsx, itemsy) tuples
        
        # Process each IO assignment for this escort
        for (target_io, itemsx, itemsy) in assignments
            # Determine which item and direction to serve
            itemid = ""
            direction = 0
            
            if !isempty(itemsx)
                direction = 1
                itemid = itemsx[1]
            elseif !isempty(itemsy)
                direction = 2
                itemid = itemsy[1]
            else
                continue  # No items for this assignment
            end
            
            if !haskey(items, itemid)
                continue
            end
            if ITEM_LOCK_ENABLED[] && !item_lock_ok(escortid, itemid)
                continue
            end

            item = items[itemid]
            if item.direction != direction
                println("Item direction and escort direction do not match for multi-IO")
                continue
            end
            
            itemx, itemy = item.coords
            escortx, escorty = escorts[escortid].coords
            iox, ioy = target_io  # Use the specific target IO for this assignment

            # Check if escort is close to its target IO
            if abs(iox - escortx) + abs(ioy - escorty) <= length(keys(items))
                checkpathformovers = true
            end
            # Check if this move would push another item close to the target IO —
            # ported from moveescorts_flow! (single-IO), was missing here. Without this,
            # an escort far from IO could move unchecked and permanently block the
            # corridor to IO once other items are nearby.
            if itemid in Iterators.flatten(values(escorts[escortid].banset)) ||
                futurecoords_closetoIO(items, itemid, escorts, escortid, direction, target_io)
                checkpathformovers = true
            end

            # Find nearest item to serve next (using target IO for this escort)
            candid, candx, candy = find_nearest_item_toitem(matrix, items, itemid, global_blockmat, target_io, direction)
            CANDID_FIX_MODE[] == 1 && (candid = "")   # mode 1: never detour toward a "candidate" item -- always head straight for the assigned one
            gapx, gapy = abs(candx - iox), abs(candy - ioy)

            escort_finalcoords = (escortx, escorty)  # Default: no movement

            # Idle-flex may have recruited an escort that isn't yet on the
            # item's line. Walk it onto that line first (perpendicular axis);
            # the normal candid logic below only runs once it's aligned.
            aligned_ok = true
            if direction == 1 && escorty != itemy
                escort_finalcoords = (escortx, itemy); aligned_ok = false
            elseif direction == 2 && escortx != itemx
                escort_finalcoords = (itemx, escorty); aligned_ok = false
            end

            if aligned_ok && direction == 1  # x-movement
                if candid == "" || candx == itemx
                    escort_finalcoords = (itemx, escorty)
                else
                    branch = ""
                    if items[candid].direction == 1 || gapx < gapy  # moving in x
                        if iox > min(itemx, candx)
                            escort_finalcoords = (max(1, candx + 1), itemy)  # IO on the right
                        elseif iox < min(itemx, candx)
                            escort_finalcoords = (max(1, candx - 1), itemy)  # IO on the left
                        end
                        branch = "push_past_cand"
                    elseif items[candid].direction == 2 || gapy <= gapx  # moving in y
                        escort_finalcoords = (candx, itemy)
                        branch = "stop_at_cand"
                    end
                    if LOG_MOVER_CANDID[]
                        push!(MOVER_CANDID_LOG, (iteration=iteration, dir="x", escortid=escortid, itemid=itemid,
                            itemx=itemx, itemy=itemy, escortx=escortx, escorty=escorty,
                            candid=candid, candx=candx, candy=candy, cand_direction=items[candid].direction,
                            gapx=gapx, gapy=gapy, branch=branch, target=escort_finalcoords))
                    end
                end

            elseif aligned_ok && direction == 2  # y-movement
                if candid == "" || candy == itemy
                    escort_finalcoords = (escortx, itemy)
                else
                    branch = ""
                    if items[candid].direction == 1 || gapx < gapy  # moving in x
                        escort_finalcoords = (itemx, candy)
                        branch = "stop_at_cand"
                    elseif items[candid].direction == 2 || gapy <= gapx  # moving in y
                        escort_finalcoords = (itemx, max(1, candy - 1))
                        branch = "push_past_cand"
                    end
                    if LOG_MOVER_CANDID[]
                        push!(MOVER_CANDID_LOG, (iteration=iteration, dir="y", escortid=escortid, itemid=itemid,
                            itemx=itemx, itemy=itemy, escortx=escortx, escorty=escorty,
                            candid=candid, candx=candx, candy=candy, cand_direction=items[candid].direction,
                            gapx=gapx, gapy=gapy, branch=branch, target=escort_finalcoords))
                    end
                end
            end
            
            # Validate movement
            if escort_finalcoords != (escortx, escorty)
                # Mirror moveescorts_flow! (single-IO): the path-check judges the
                # uncapped, potentially multi-cell escort_finalcoords, but under LM
                # mode move_escort! only steps one cell toward it, so the check would
                # ban safe 1-cell moves (e.g. an escort on the IO pushing an adjacent
                # item into it — path_to_io_exists_if sees the IO cell "blocked" by
                # the item's future position and returns false). Bypass it in LM mode.
                if checkpathformovers && (!UNIT_STEP[] || LM_PATHCHECK[])   # path-to-IO check: BM always, LM when LM_PATHCHECK[]
                    # Use IO-specific blockmat for this assignment
                    
                    itemscoords = generatefuturecoords_fincoord(items, escorts, direction, escortid, escort_finalcoords, matrix, target_io)
                    samecoords = direction == 2 ?
                        filter(x -> x[1] == itemx && x[2] >= min(itemy, escorty) && x[2] <= max(itemy, escorty), itemscoords) :
                        filter(x -> x[2] == itemy && x[1] >= min(itemx, escortx) && x[1] <= max(itemx, escortx), itemscoords)
                    
                    if !isempty(samecoords)
                        minDist = minimum([abs(iox - coord[1]) + abs(ioy - coord[2]) for coord in samecoords])
                        if minDist > length(keys(items)) || path_to_io_exists_if(matrix, itemscoords, target_io)
                            # Safe to move
                            TRACE_ESC[] == escortid && println("t=$iteration [MOVER] $escortid ($escortx,$escorty) -> $escort_finalcoords  item=$itemid dir=$direction io=$target_io")
                            push!(escorts[escortid].tabu, (escortx, escorty))
                            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                            updateblockmat_e!(global_blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2])
                            escorts[escortid].lastmoved = iteration
                        else
                            # Ban this escort from moving this item next iteration
                            if !haskey(escorts[escortid].banset, iteration + 1)
                                escorts[escortid].banset[iteration + 1] = [itemid]
                            else
                                push!(escorts[escortid].banset[iteration + 1], itemid)
                            end
                            updateblockmat_e!(global_blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2], val=0)
                        end
                    end
                else
                    # No path check needed, move directly
                    TRACE_ESC[] == escortid && println("t=$iteration [MOVER-nocheck] $escortid ($escortx,$escorty) -> $escort_finalcoords  item=$itemid dir=$direction io=$target_io")
                    push!(escorts[escortid].tabu, (escortx, escorty))
                    moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                    
                    updateblockmat_e!(global_blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2])
                    escorts[escortid].lastmoved = iteration
                end
            end
        end
    end
    
# ── NON-MOVERS ──────────────────────────────────────────────────────────
    diagonal_size = sqrt(size(matrix, 1)^2 + size(matrix, 2)^2)

    # Urgency: each item is checked against its own assigned IO (closest one if multiple)
    urgentcustomers = filter(customer_id -> begin
        item = items[customer_id]
        assigned = get(item_to_ios, customer_id, all_ios)
        iox, ioy = argmin(io -> abs(io[1] - item.coords[1]) + abs(io[2] - item.coords[2]), assigned)
        floor(Int, iteration + (abs(iox - item.coords[1]) + item.coords[2]) * 1.5) >= item.deadline ||
        (iteration - item.tes) + (abs(item.coords[1] - iox) + abs(item.coords[2] - ioy)) > diagonal_size
    end, keys(items))

    nonmovers = setdiff(keys(escorts), moverescortids)
    if ITEM_LOCK_ENABLED[]
        reclaimed = filter(e -> !mover_matches_lock(e, global_escort_items), moverescortids)
        nonmovers = union(nonmovers, reclaimed)
        if TRACE_ESC[] != "" && TRACE_ESC[] in moverescortids
            println("    t=$iteration RECLAIM-CHECK $(TRACE_ESC[]) assignments=$(get(global_escort_items, TRACE_ESC[], [])) matches_lock=$(mover_matches_lock(TRACE_ESC[], global_escort_items)) reclaimed=$(TRACE_ESC[] in reclaimed) in_nonmovers=$(TRACE_ESC[] in nonmovers)")
        end
    end
    FREEZE_ENABLED[] && (nonmovers = Set(filter(e -> escort_active(e, iteration), collect(nonmovers))))

    # Sort: escorts farther from all IOs go first (they need to reposition more urgently),
    # break ties by empty space ahead in y
    nonmovers = sort(collect(nonmovers), by = escortid -> begin
        esc_x, esc_y = escorts[escortid].coords
        empty_spaces = 0
        for y in esc_y-1:-1:1
            if global_blockmat[esc_x, y] == 1 || haskey(items, matrix[esc_x, y]) || haskey(escorts, matrix[esc_x, y])
                break
            end
            empty_spaces += 1
        end
        dist_to_nearest_io = minimum(io -> abs(io[1] - esc_x) + abs(io[2] - esc_y), all_ios)
        return (-dist_to_nearest_io, empty_spaces)
    end, rev=false)

    # Items per IO zone (nearest-IO). Also drives the smart/dumb freeroam split below.
    io_item_counts = Dict(io => count(
        id -> argmin(pio -> abs(pio[1] - items[id].coords[1]) + abs(pio[2] - items[id].coords[2]), all_ios) == io,
        keys(items)) for io in all_ios)

    # LM multi-IO park gate: an idle escort may only move (directserve / urgserve /
    # freeroam) if its zone IO still has items, or is within PARK_GATE_RADIUS[]
    # IO-zones of one that does. Escorts further than that from any demand hold —
    # marching them across other IO zones toward the busy one is pure wasted
    # moves. Recomputed every iteration, so each delivery that empties a zone
    # parks that zone's escorts on the spot. Gated to >3 IOs and >1 item left.
    parked = Set{String}()   # visible to the idle-flex redirect below, which may un-park
    if UNIT_STEP[] && length(all_ios) > 3 && length(items) > 1
        ios_x = sort(collect(all_ios), by = io -> (io[1], io[2]))
        has_demand = Dict(io => io_item_counts[io] > 0 for io in ios_x)
        for cio in values(COMMITTED_IO)
            haskey(has_demand, cio) && (has_demand[cio] = true)
        end
        movable = Dict(io => has_demand[io] for io in ios_x)
        for (i, io) in enumerate(ios_x)
            has_demand[io] || continue
            for d in 1:PARK_GATE_RADIUS[]
                i - d >= 1             && (movable[ios_x[i-d]] = true)
                i + d <= length(ios_x) && (movable[ios_x[i+d]] = true)
            end
        end
        for eid in nonmovers
            # A locked escort with a still-undelivered permitted item always has
            # real work regardless of its current zone's demand -- never park it.
            ITEM_LOCK_ENABLED[] && any(it -> haskey(items, it), get(ESCORT_ITEM_LOCK, eid, Set{String}())) && continue
            ex, ey = escorts[eid].coords
            zone = argmin(io -> abs(io[1] - ex) + abs(io[2] - ey), ios_x)
            movable[zone] || push!(parked, eid)
        end
        nonmovers = filter(e -> !(e in parked), nonmovers)
        if iteration == DBG_ITER[]
            println("t=$iteration PARKGATE  movable_zones=$([io for io in ios_x if movable[io]])")
            println("    parked (x>=50): $([ (e, escorts[e].coords) for e in parked if escorts[e].coords[1] >= 50 ])")
            println("    survivors (x>=50): $([ (e, escorts[e].coords) for e in nonmovers if escorts[e].coords[1] >= 50 ])")
        end
    end

    usedescorts = String[]
    for escortid in nonmovers
        esc_x, esc_y = escorts[escortid].coords
        if global_blockmat[esc_x, esc_y] == 1
            continue
        end
        moved, escort_finalcoords = directserve_flow_multi_io!(iteration, matrix, items, escorts,
                                                                escortid, urgentcustomers,
                                                                global_blockmat, item_to_ios, all_ios)
        if UNIT_STEP[] && moved && escort_finalcoords != (esc_x, esc_y) &&
           _lm_realized_step((esc_x, esc_y), escort_finalcoords) in escorts[escortid].tabu
            moved = false   # single step would backtrack onto a just-vacated cell
        end
        if moved && escort_finalcoords != (esc_x, esc_y)
            #println("iter $iteration: moved $escortid from ($esc_x, $esc_y) to $escort_finalcoords")
            TRACE_ESC[] == escortid && println("t=$iteration [DIRECTSERVE] $escortid ($esc_x,$esc_y) -> $escort_finalcoords")
            push!(escorts[escortid].tabu, (esc_x, esc_y))
            push!(usedescorts, escortid)
            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
            updateblockmat_e!(global_blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
            escorts[escortid].lastmoved = iteration
        end
    end
    nonmovers = setdiff(nonmovers, usedescorts)
    usedescorts = String[]
    if !isempty(urgentcustomers)
        # Build urgmats using each item's own assigned IO, not a single global IO
        urgentmatrixes = urgmats_multi_io(items, escorts, global_blockmat, matrix, 
                                           urgentcustomers, item_to_ios, all_ios)
        for escortid in nonmovers
            esc_x, esc_y = escorts[escortid].coords
            if global_blockmat[esc_x, esc_y] == 1
                continue
            end
            moved, escort_finalcoords = urgserve_multi_io!(iteration, matrix, items, escorts,
                                                            escortid, urgentmatrixes,
                                                            item_to_ios, all_ios)
            if UNIT_STEP[] && moved && escort_finalcoords != (esc_x, esc_y) &&
               _lm_realized_step((esc_x, esc_y), escort_finalcoords) in escorts[escortid].tabu
                moved = false   # single step would backtrack onto a just-vacated cell
            end
            if moved && escort_finalcoords != (esc_x, esc_y)
                #println("iter $iteration: moved $escortid from ($esc_x, $esc_y) to ($escort_finalcoords)")
                TRACE_ESC[] == escortid && println("t=$iteration [URGSERVE] $escortid ($esc_x,$esc_y) -> $escort_finalcoords")
                push!(escorts[escortid].tabu, (esc_x, esc_y))
                push!(usedescorts, escortid)
                moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                updateblockmat_e!(global_blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                updateurgmats_e!(urgentmatrixes, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                escorts[escortid].lastmoved = iteration
            end
        end
    end
    nonmovers = setdiff(nonmovers, usedescorts)

  # ── BALANCING + FREEROAM ─────────────────────────────────────────────────
    # Count escorts "belonging" to each IO: nearest IO wins
    io_escort_counts = Dict(io => 0 for io in all_ios)
    io_escort_members = Dict(io => [] for io in all_ios)
    for escortid in nonmovers
        esc_x, esc_y = escorts[escortid].coords
        nearest_io = argmin(io -> abs(io[1] - esc_x) + abs(io[2] - esc_y), all_ios)
        io_escort_counts[nearest_io] += 1
        push!(io_escort_members[nearest_io], escortid)
    end

    # How many escorts should each IO ideally have
    target_per_io = length(nonmovers) / length(all_ios)

    # Build a list of (escortid, target_io) for each nonmover:
    # - escorts in overpopulated zones get assigned to the nearest underpopulated IO
    # - all others stay in their current zone
    escort_target_ios = Dict{eltype(nonmovers), Any}()

    # Sort IOs: overpopulated ones "donate" escorts to underpopulated ones
    sorted_ios_by_excess = sort(all_ios, by = io -> -io_escort_counts[io])  # most populated first
    
    # Build a transfer list: (escortid, destination_io)
    # For each overpopulated IO, take the escorts farthest from it (most "transferable")
    # and assign them to the most underpopulated IO
    for escortid in nonmovers
        esc_x, esc_y = escorts[escortid].coords
        nearest_io = argmin(io -> abs(io[1] - esc_x) + abs(io[2] - esc_y), all_ios)
        escort_target_ios[escortid] = nearest_io  # default: stay in own zone
    end

    # Transfer excess escorts from overpopulated IOs to underpopulated IOs
    for io in sorted_ios_by_excess
        remaining_excess = io_escort_counts[io] - ceil(Int, target_per_io)
        if remaining_excess <= 0; continue; end

        # Only consider escorts still assigned to this IO (not already transferred away)
        not_yet_transferred = filter(eid -> escort_target_ios[eid] == io, io_escort_members[io])

        # Sort by closeness to target — but target may change per escort, so sort by
        # distance to the centroid of all underpopulated IOs as a proxy
        underpopulated = filter(io2 -> io_escort_counts[io2] < floor(Int, target_per_io), all_ios)
        if isempty(underpopulated); continue; end
        centroid_x = mean(io2[1] for io2 in underpopulated)
        centroid_y = mean(io2[2] for io2 in underpopulated)
        transferable = sort(not_yet_transferred,
            by = eid -> abs(centroid_x - escorts[eid].coords[1]) + abs(centroid_y - escorts[eid].coords[2]))

        for eid in transferable
            if remaining_excess <= 0; break; end
            # Recompute the most underpopulated IO for each individual escort transfer
            target_io = argmin(io2 -> io_escort_counts[io2], all_ios)
            if io_escort_counts[target_io] >= ceil(Int, target_per_io); break; end

            escort_target_ios[eid] = target_io
            io_escort_counts[io] -= 1
            io_escort_counts[target_io] += 1
            remaining_excess -= 1
        end
    end

    # LM only: if an IO has no items targeting it this iteration, don't let idle
    # escorts freeroam toward it — redirect each such escort to the nearest IO that
    # does have demand. Stops the lone escort parking on an empty IO while items
    # wait at the other one. (BM keeps its balancing-only targeting.)
    if UNIT_STEP[]
        demand_ios = Set{Tuple{Int,Int}}()
        for ios in values(item_to_ios), io in ios
            push!(demand_ios, (io[1], io[2]))
        end
        for cio in values(COMMITTED_IO)
            push!(demand_ios, cio)
        end
        if !isempty(demand_ios)
            for escortid in nonmovers
                cur = escort_target_ios[escortid]
                if !((cur[1], cur[2]) in demand_ios)
                    ex, ey = escorts[escortid].coords
                    escort_target_ios[escortid] = argmin(io -> abs(io[1] - ex) + abs(io[2] - ey),
                                                         collect(demand_ios))
                end
            end
        end
    end

    # Item lock: a locked escort's real job is its permitted item(s), not
    # whatever the zone-balancing logic above decided -- override target_io to
    # head straight for the IO of its nearest still-undelivered permitted item.
    if ITEM_LOCK_ENABLED[]
        for escortid in nonmovers
            locked = get(ESCORT_ITEM_LOCK, escortid, Set{String}())
            active = [it for it in locked if haskey(items, it)]
            isempty(active) && continue
            esc_x, esc_y = escorts[escortid].coords
            nearest_item = argmin(it -> abs(items[it].coords[1]-esc_x)+abs(items[it].coords[2]-esc_y), active)
            assigned = get(item_to_ios, nearest_item, all_ios)
            escort_target_ios[escortid] = argmin(io -> abs(io[1]-items[nearest_item].coords[1])+abs(io[2]-items[nearest_item].coords[2]), assigned)
        end
    end

    # Idle-flex redirect: for any item that has been stagnant a long time,
    # pull its 2 nearest free (nonmover) escorts toward it by pointing their
    # freeroam target at the item's IO, so they physically close the distance
    # until the relaxed alignment tolerance lets the assignment recruit them.
    if IDLE_FLEX[]
        stuck = [k for k in keys(items) if get(ITEM_IDLE, k, 0) >= IDLE_FLEX_REDIRECT[]]
        _dbg = iteration == DBG_ITER[]
        _dbg && println("t=$iteration REDIRECT  stuck=$stuck  nonmovers=$(length(nonmovers))  parked=$(length(parked))")
        if !isempty(stuck)
            used = Set{String}()
            # Draw candidates from free escorts AND parked ones -- extending the
            # reach for a long-stuck item is allowed to un-park escorts.
            for k in stuck
                ix, iy = items[k].coords
                iio = argmin(io -> abs(io[1]-ix)+abs(io[2]-iy), get(item_to_ios, k, all_ios))
                pool = [e for e in Iterators.flatten((collect(nonmovers), collect(parked))) if !(e in used)]
                cands = sort(pool, by = e -> abs(escorts[e].coords[1]-ix)+abs(escorts[e].coords[2]-iy))
                for e in cands[1:min(2, length(cands))]
                    if e in parked
                        delete!(parked, e)
                        e in nonmovers || push!(nonmovers, e)
                    end
                    escort_target_ios[e] = iio
                    push!(used, e)
                    _dbg && println("    redirect $e $(escorts[e].coords) -> io $iio  (for stuck $k @ ($ix,$iy))")
                end
            end
        end
    end

    # How many escorts per IO should use freeroam! first (items count, or half if no items)
    io_smart_remaining = Dict(io => let e = io_escort_counts[io], n = io_item_counts[io]
        e > n ? (n == 0 ? fld(e, 2) : n) : e
    end for io in all_ios)

    # Freeroam: each escort uses its assigned target_io as the IO argument
    _dbgfr = iteration == DBG_ITER[]
    for escortid in nonmovers
        esc_x, esc_y = escorts[escortid].coords
        if global_blockmat[esc_x, esc_y] == 1
            _dbgfr && esc_x >= 55 && println("    FR $escortid ($esc_x,$esc_y) SKIP blockmat==1")
            continue
        end

        target_io = escort_target_ios[escortid]

        _tr_phase = ""
        if io_smart_remaining[target_io] > 0
            io_smart_remaining[target_io] -= 1
            _tr_phase = "FREEROAM"
            moved, escort_finalcoords = freeroam!(iteration, matrix, items, escorts,
                                                   escortid, global_blockmat, target_io)
        else
            _tr_phase = "FREEROAM_DUMB"
            moved, escort_finalcoords = freeroam_dumb!(iteration, matrix, items, escorts,
                                                        escortid, global_blockmat, target_io)
        end

        _dbgfr && (esc_x >= 55 || target_io == (61,1)) && println("    FR $escortid ($esc_x,$esc_y) target_io=$target_io  $_tr_phase moved=$moved -> $escort_finalcoords")

        if moved && escort_finalcoords != (esc_x, esc_y)
            TRACE_ESC[] == escortid && println("t=$iteration [$_tr_phase] $escortid ($esc_x,$esc_y) -> $escort_finalcoords  target_io=$target_io")
            push!(escorts[escortid].tabu, (esc_x, esc_y))
            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
            updateblockmat_e!(global_blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
            escorts[escortid].lastmoved = iteration
        end
    end


    return (moved_any > 0)
end

"""
GRASP version of moveescorts_flow_multi_io! — mirrors the same randomization
applied by moveescorts_flow_r! to the single-IO case:
  - movers: random pick between x/y direction (when both available) and random
    item within that direction's list, instead of always itemsx[1]/itemsy[1]
  - nonmovers: GRASP-randomized sort (sort_nonmovers_multi_io_grasp) instead of
    the deterministic sort
  - direct-serve / urgent-serve use their GRASP/shuffled counterparts
The IO-balancing + freeroam!/freeroam_dumb! tail is left unchanged — it's
inherently multi-IO-specific bookkeeping with no single-IO randomized analog
to mirror (single-IO's randomized tail uses cooperative_freeroam!, which has
no multi-IO equivalent).
"""
function moveescorts_flow_multi_io_r!(iteration, matrix, items, escorts, global_blockmat,
                                     global_escort_items, all_ios, item_to_ios)
    moverescortids = collect(keys(global_escort_items))
    moved_any = 0
    checkpathformovers = false

    # GRASP version of moveescorts_flow!'s nearest-escort-to-its-item mover
    # ordering, adapted for the multi-IO (io, itemsx, itemsy) assignment shape:
    # the distance metric is the closest item across all of an escort's
    # assignments. See sort_movers_by_item_distance_grasp docstring.
    let
        function dist_key(escortid)
            esc_x, esc_y = escorts[escortid].coords
            best = Inf
            for (_, itemsx, itemsy) in global_escort_items[escortid]
                for k in Iterators.flatten((itemsx, itemsy))
                    d = abs(esc_x - items[k].coords[1]) + abs(esc_y - items[k].coords[2])
                    d < best && (best = d)
                end
            end
            return best
        end

        pool = copy(moverescortids)
        result = String[]
        while !isempty(pool)
            ranked = sort(pool, by = dist_key)
            idx = grasp_rank_pick(Random.default_rng(), length(ranked), 0.8)
            chosen = ranked[idx]
            push!(result, chosen)
            filter!(e -> e != chosen, pool)
        end
        moverescortids = result
    end

    for escortid in moverescortids
        if !haskey(global_escort_items, escortid)
            continue
        end

        assignments = global_escort_items[escortid]  # List of (io, itemsx, itemsy) tuples

        # Process each IO assignment for this escort
        for (target_io, itemsx, itemsy) in assignments
            # Determine which item and direction to serve
            itemid = ""
            direction = 0

            # RANDOMIZATION: randomize which direction and which item to serve,
            # instead of always itemsx[1]/itemsy[1]
            if !isempty(itemsx) && !isempty(itemsy)
                if rand(Random.default_rng(), Bool)
                    direction = 1; itemid = rand(Random.default_rng(), itemsx)
                else
                    direction = 2; itemid = rand(Random.default_rng(), itemsy)
                end
            elseif !isempty(itemsx)
                direction = 1; itemid = rand(Random.default_rng(), itemsx)
            elseif !isempty(itemsy)
                direction = 2; itemid = rand(Random.default_rng(), itemsy)
            else
                continue  # No items for this assignment
            end

            if !haskey(items, itemid)
                continue
            end

            item = items[itemid]
            if item.direction != direction
                println("Item direction and escort direction do not match for multi-IO")
                continue
            end

            itemx, itemy = item.coords
            escortx, escorty = escorts[escortid].coords
            iox, ioy = target_io  # Use the specific target IO for this assignment

            # Check if escort is close to its target IO
            if abs(iox - escortx) + abs(ioy - escorty) <= length(keys(items))
                checkpathformovers = true
            end
            # Check if this move would push another item close to the target IO —
            # ported from moveescorts_flow_r! (single-IO); same fix as moveescorts_flow_multi_io!.
            if itemid in Iterators.flatten(values(escorts[escortid].banset)) ||
                futurecoords_closetoIO(items, itemid, escorts, escortid, direction, target_io)
                checkpathformovers = true
            end

            # Find nearest item to serve next (using target IO for this escort)
            candid, candx, candy = find_nearest_item_toitem(matrix, items, itemid, global_blockmat, target_io, direction)
            CANDID_FIX_MODE[] == 1 && (candid = "")   # mode 1: never detour toward a "candidate" item -- always head straight for the assigned one
            gapx, gapy = abs(candx - iox), abs(candy - ioy)

            escort_finalcoords = (escortx, escorty)  # Default: no movement

            if direction == 1  # x-movement
                if candid == "" || candx == itemx
                    escort_finalcoords = (itemx, escorty)
                else
                    if items[candid].direction == 1 || gapx < gapy  # moving in x
                        if iox > min(itemx, candx)
                            escort_finalcoords = (max(1, candx + 1), itemy)  # IO on the right
                        elseif iox < min(itemx, candx)
                            escort_finalcoords = (max(1, candx - 1), itemy)  # IO on the left
                        end
                    elseif items[candid].direction == 2 || gapy <= gapx  # moving in y
                        escort_finalcoords = (candx, itemy)
                    end
                end

            elseif direction == 2  # y-movement
                if candid == "" || candy == itemy
                    escort_finalcoords = (escortx, itemy)
                else
                    if items[candid].direction == 1 || gapx < gapy  # moving in x
                        escort_finalcoords = (itemx, candy)
                    elseif items[candid].direction == 2 || gapy <= gapx  # moving in y
                        escort_finalcoords = (itemx, max(1, candy - 1))
                    end
                end
            end

            # Validate movement
            if escort_finalcoords != (escortx, escorty)
                # Mirror moveescorts_flow! (single-IO): the path-check judges the
                # uncapped, potentially multi-cell escort_finalcoords, but under LM
                # mode move_escort! only steps one cell toward it, so the check would
                # ban safe 1-cell moves (e.g. an escort on the IO pushing an adjacent
                # item into it — path_to_io_exists_if sees the IO cell "blocked" by
                # the item's future position and returns false). Bypass it in LM mode.
                if checkpathformovers && !UNIT_STEP[]
                    # Use IO-specific blockmat for this assignment

                    itemscoords = generatefuturecoords_fincoord(items, escorts, direction, escortid, escort_finalcoords, matrix, target_io)
                    samecoords = direction == 2 ?
                        filter(x -> x[1] == itemx && x[2] >= min(itemy, escorty) && x[2] <= max(itemy, escorty), itemscoords) :
                        filter(x -> x[2] == itemy && x[1] >= min(itemx, escortx) && x[1] <= max(itemx, escortx), itemscoords)

                    if !isempty(samecoords)
                        minDist = minimum([abs(iox - coord[1]) + abs(ioy - coord[2]) for coord in samecoords])

                        if minDist > length(keys(items)) || path_to_io_exists_if(matrix, itemscoords, target_io)
                            # Safe to move
                            TRACE_ESC[] == escortid && println("t=$iteration [MOVER] $escortid ($escortx,$escorty) -> $escort_finalcoords  item=$itemid dir=$direction io=$target_io")
                            push!(escorts[escortid].tabu, (escortx, escorty))
                            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                            updateblockmat_e!(global_blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2])
                            escorts[escortid].lastmoved = iteration
                        else
                            # Ban this escort from moving this item next iteration
                            if !haskey(escorts[escortid].banset, iteration + 1)
                                escorts[escortid].banset[iteration + 1] = [itemid]
                            else
                                push!(escorts[escortid].banset[iteration + 1], itemid)
                            end
                            updateblockmat_e!(global_blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2], val=0)
                        end
                    end
                else
                    # No path check needed, move directly
                    TRACE_ESC[] == escortid && println("t=$iteration [MOVER-nocheck] $escortid ($escortx,$escorty) -> $escort_finalcoords  item=$itemid dir=$direction io=$target_io")
                    push!(escorts[escortid].tabu, (escortx, escorty))
                    moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)

                    updateblockmat_e!(global_blockmat, escortx, escorty, escort_finalcoords[1], escort_finalcoords[2])
                    escorts[escortid].lastmoved = iteration
                end
            end
        end
    end

# ── NON-MOVERS ──────────────────────────────────────────────────────────
    diagonal_size = sqrt(size(matrix, 1)^2 + size(matrix, 2)^2)

    # Urgency: each item is checked against its own assigned IO (closest one if multiple)
    urgentcustomers = filter(customer_id -> begin
        item = items[customer_id]
        assigned = get(item_to_ios, customer_id, all_ios)
        iox, ioy = argmin(io -> abs(io[1] - item.coords[1]) + abs(io[2] - item.coords[2]), assigned)
        floor(Int, iteration + (abs(iox - item.coords[1]) + item.coords[2]) * 1.5) >= item.deadline ||
        (iteration - item.tes) + (abs(item.coords[1] - iox) + abs(item.coords[2] - ioy)) > diagonal_size
    end, keys(items))

    nonmovers = setdiff(keys(escorts), moverescortids)
    FREEZE_ENABLED[] && (nonmovers = Set(filter(e -> escort_active(e, iteration), collect(nonmovers))))
    parked = Set{String}()   # GRASP twin has no park gate; kept empty so the shared redirect block below compiles

    # RANDOMIZATION: GRASP sort (α=0.3) instead of the deterministic sort
    nonmovers = sort_nonmovers_multi_io_grasp(nonmovers, escorts, items, matrix, global_blockmat, all_ios, 0.3)

    usedescorts = String[]
    for escortid in nonmovers
        esc_x, esc_y = escorts[escortid].coords
        if global_blockmat[esc_x, esc_y] == 1
            continue
        end
        moved, escort_finalcoords = directserve_flow_multi_io_r!(iteration, matrix, items, escorts,
                                                                escortid, urgentcustomers,
                                                                global_blockmat, item_to_ios, all_ios)
        if UNIT_STEP[] && moved && escort_finalcoords != (esc_x, esc_y) &&
           _lm_realized_step((esc_x, esc_y), escort_finalcoords) in escorts[escortid].tabu
            moved = false
        end

        if moved && escort_finalcoords != (esc_x, esc_y)
            push!(escorts[escortid].tabu, (esc_x, esc_y))
            push!(usedescorts, escortid)
            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
            updateblockmat_e!(global_blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
            escorts[escortid].lastmoved = iteration
        end
    end
    nonmovers = setdiff(nonmovers, usedescorts)
    usedescorts = String[]
    if !isempty(urgentcustomers)
        # Build urgmats using each item's own assigned IO, not a single global IO
        urgentmatrixes = urgmats_multi_io(items, escorts, global_blockmat, matrix,
                                           urgentcustomers, item_to_ios, all_ios)
        for escortid in nonmovers
            esc_x, esc_y = escorts[escortid].coords
            if global_blockmat[esc_x, esc_y] == 1
                continue
            end
            moved, escort_finalcoords = urgserve_multi_io_r!(iteration, matrix, items, escorts,
                                                            escortid, urgentmatrixes,
                                                            item_to_ios, all_ios)
            if UNIT_STEP[] && moved && escort_finalcoords != (esc_x, esc_y) &&
               _lm_realized_step((esc_x, esc_y), escort_finalcoords) in escorts[escortid].tabu
                moved = false
            end
            if moved && escort_finalcoords != (esc_x, esc_y)
                push!(escorts[escortid].tabu, (esc_x, esc_y))
                push!(usedescorts, escortid)
                moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
                updateblockmat_e!(global_blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                updateurgmats_e!(urgentmatrixes, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
                escorts[escortid].lastmoved = iteration
            end
        end
    end
    nonmovers = setdiff(nonmovers, usedescorts)

  # ── BALANCING + FREEROAM (unchanged — multi-IO-specific, no single-IO analog) ──
    # Count escorts "belonging" to each IO: nearest IO wins
    io_escort_counts = Dict(io => 0 for io in all_ios)
    io_escort_members = Dict(io => [] for io in all_ios)
    for escortid in nonmovers
        esc_x, esc_y = escorts[escortid].coords
        nearest_io = argmin(io -> abs(io[1] - esc_x) + abs(io[2] - esc_y), all_ios)
        io_escort_counts[nearest_io] += 1
        push!(io_escort_members[nearest_io], escortid)
    end

    # How many escorts should each IO ideally have
    target_per_io = length(nonmovers) / length(all_ios)

    # Build a list of (escortid, target_io) for each nonmover:
    # - escorts in overpopulated zones get assigned to the nearest underpopulated IO
    # - all others stay in their current zone
    escort_target_ios = Dict{eltype(nonmovers), Any}()

    # Sort IOs: overpopulated ones "donate" escorts to underpopulated ones
    sorted_ios_by_excess = sort(all_ios, by = io -> -io_escort_counts[io])  # most populated first

    # Build a transfer list: (escortid, destination_io)
    # For each overpopulated IO, take the escorts farthest from it (most "transferable")
    # and assign them to the most underpopulated IO
    for escortid in nonmovers
        esc_x, esc_y = escorts[escortid].coords
        nearest_io = argmin(io -> abs(io[1] - esc_x) + abs(io[2] - esc_y), all_ios)
        escort_target_ios[escortid] = nearest_io  # default: stay in own zone
    end

    # Transfer excess escorts from overpopulated IOs to underpopulated IOs
    for io in sorted_ios_by_excess
        remaining_excess = io_escort_counts[io] - ceil(Int, target_per_io)
        if remaining_excess <= 0; continue; end

        # Only consider escorts still assigned to this IO (not already transferred away)
        not_yet_transferred = filter(eid -> escort_target_ios[eid] == io, io_escort_members[io])

        # Sort by closeness to target — but target may change per escort, so sort by
        # distance to the centroid of all underpopulated IOs as a proxy
        underpopulated = filter(io2 -> io_escort_counts[io2] < floor(Int, target_per_io), all_ios)
        if isempty(underpopulated); continue; end
        centroid_x = mean(io2[1] for io2 in underpopulated)
        centroid_y = mean(io2[2] for io2 in underpopulated)
        transferable = sort(not_yet_transferred,
            by = eid -> abs(centroid_x - escorts[eid].coords[1]) + abs(centroid_y - escorts[eid].coords[2]))

        for eid in transferable
            if remaining_excess <= 0; break; end
            # Recompute the most underpopulated IO for each individual escort transfer
            target_io = argmin(io2 -> io_escort_counts[io2], all_ios)
            if io_escort_counts[target_io] >= ceil(Int, target_per_io); break; end

            escort_target_ios[eid] = target_io
            io_escort_counts[io] -= 1
            io_escort_counts[target_io] += 1
            remaining_excess -= 1
        end
    end

    # LM only: don't freeroam idle escorts toward an IO with no items targeting it
    # (see moveescorts_flow_multi_io! for rationale).
    if UNIT_STEP[]
        demand_ios = Set{Tuple{Int,Int}}()
        for ios in values(item_to_ios), io in ios
            push!(demand_ios, (io[1], io[2]))
        end
        for cio in values(COMMITTED_IO)
            push!(demand_ios, cio)
        end
        if !isempty(demand_ios)
            for escortid in nonmovers
                cur = escort_target_ios[escortid]
                if !((cur[1], cur[2]) in demand_ios)
                    ex, ey = escorts[escortid].coords
                    escort_target_ios[escortid] = argmin(io -> abs(io[1] - ex) + abs(io[2] - ey),
                                                         collect(demand_ios))
                end
            end
        end
    end

    #ORIGINAL: freeroam!/freeroam_dumb! per escort, using its assigned target_io.
    # Kept for reference/rollback — replaced below by grouped cooperative_freeroam! calls.

    # Count items per IO zone (to decide smart vs dumb within each zone)
    io_item_counts = Dict(io => count(
        id -> argmin(pio -> abs(pio[1] - items[id].coords[1]) + abs(pio[2] - items[id].coords[2]), all_ios) == io,
        keys(items)) for io in all_ios)

    # Item lock: a locked escort's real job is its permitted item(s), not
    # whatever the zone-balancing logic above decided -- override target_io to
    # head straight for the IO of its nearest still-undelivered permitted item.
    if ITEM_LOCK_ENABLED[]
        for escortid in nonmovers
            locked = get(ESCORT_ITEM_LOCK, escortid, Set{String}())
            active = [it for it in locked if haskey(items, it)]
            isempty(active) && continue
            esc_x, esc_y = escorts[escortid].coords
            nearest_item = argmin(it -> abs(items[it].coords[1]-esc_x)+abs(items[it].coords[2]-esc_y), active)
            assigned = get(item_to_ios, nearest_item, all_ios)
            escort_target_ios[escortid] = argmin(io -> abs(io[1]-items[nearest_item].coords[1])+abs(io[2]-items[nearest_item].coords[2]), assigned)
        end
    end

    # Idle-flex redirect: for any item that has been stagnant a long time,
    # pull its 2 nearest free (nonmover) escorts toward it by pointing their
    # freeroam target at the item's IO, so they physically close the distance
    # until the relaxed alignment tolerance lets the assignment recruit them.
    if IDLE_FLEX[]
        stuck = [k for k in keys(items) if get(ITEM_IDLE, k, 0) >= IDLE_FLEX_REDIRECT[]]
        _dbg = iteration == DBG_ITER[]
        _dbg && println("t=$iteration REDIRECT  stuck=$stuck  nonmovers=$(length(nonmovers))  parked=$(length(parked))")
        if !isempty(stuck)
            used = Set{String}()
            # Draw candidates from free escorts AND parked ones -- extending the
            # reach for a long-stuck item is allowed to un-park escorts.
            for k in stuck
                ix, iy = items[k].coords
                iio = argmin(io -> abs(io[1]-ix)+abs(io[2]-iy), get(item_to_ios, k, all_ios))
                pool = [e for e in Iterators.flatten((collect(nonmovers), collect(parked))) if !(e in used)]
                cands = sort(pool, by = e -> abs(escorts[e].coords[1]-ix)+abs(escorts[e].coords[2]-iy))
                for e in cands[1:min(2, length(cands))]
                    if e in parked
                        delete!(parked, e)
                        e in nonmovers || push!(nonmovers, e)
                    end
                    escort_target_ios[e] = iio
                    push!(used, e)
                    _dbg && println("    redirect $e $(escorts[e].coords) -> io $iio  (for stuck $k @ ($ix,$iy))")
                end
            end
        end
    end

    # How many escorts per IO should use freeroam! first (items count, or half if no items)
    io_smart_remaining = Dict(io => let e = io_escort_counts[io], n = io_item_counts[io]
        e > n ? (n == 0 ? fld(e, 2) : n) : e
    end for io in all_ios)

    # Freeroam: each escort uses its assigned target_io as the IO argument
    _dbgfr = iteration == DBG_ITER[]
    for escortid in nonmovers
        esc_x, esc_y = escorts[escortid].coords
        if global_blockmat[esc_x, esc_y] == 1
            _dbgfr && esc_x >= 55 && println("    FR $escortid ($esc_x,$esc_y) SKIP blockmat==1")
            continue
        end

        target_io = escort_target_ios[escortid]

        _tr_phase = ""
        if io_smart_remaining[target_io] > 0
            io_smart_remaining[target_io] -= 1
            _tr_phase = "FREEROAM"
            moved, escort_finalcoords = freeroam!(iteration, matrix, items, escorts,
                                                   escortid, global_blockmat, target_io)
        else
            _tr_phase = "FREEROAM_DUMB"
            moved, escort_finalcoords = freeroam_dumb!(iteration, matrix, items, escorts,
                                                        escortid, global_blockmat, target_io)
        end

        _dbgfr && (esc_x >= 55 || target_io == (61,1)) && println("    FR $escortid ($esc_x,$esc_y) target_io=$target_io  $_tr_phase moved=$moved -> $escort_finalcoords")

        if moved && escort_finalcoords != (esc_x, esc_y)
            TRACE_ESC[] == escortid && println("t=$iteration [$_tr_phase] $escortid ($esc_x,$esc_y) -> $escort_finalcoords  target_io=$target_io")
            push!(escorts[escortid].tabu, (esc_x, esc_y))
            moved_any += move_escort!(matrix, items, escorts, escortid, escort_finalcoords)
            updateblockmat_e!(global_blockmat, esc_x, esc_y, escort_finalcoords[1], escort_finalcoords[2])
            escorts[escortid].lastmoved = iteration
        end
    end
   

 #=    # NEW: group nonmovers by their assigned target IO (escort_target_ios, from the
    # balancing step above) and let cooperative_freeroam! jointly plan each IO's group —
    # mirrors how moveescorts_flow_r! uses cooperative_freeroam! for the single-IO case.
    for io in all_ios
        io_group = [eid for eid in nonmovers
                    if escort_target_ios[eid] == io && global_blockmat[escorts[eid].coords[1], escorts[eid].coords[2]] != 1]
        moved_any += cooperative_freeroam!(iteration, matrix, items, escorts, io_group, global_blockmat, io)
    end =#

    return (moved_any > 0)
end


candid_aligned(a, b) = CANDID_FIX_MODE[] == 0 ? true : CANDID_FIX_MODE[] == 3 ? abs(a - b) <= 1 : a == b

function find_nearest_item_toitem(matrix, items, itemid, blockmat, IO, direction)
    item = items[itemid]
    itemx, itemy = item.coords
    nearestitemx = size(matrix, 1)+1
    nearestitemy = size(matrix, 2)+1
    nearestitemid = ""
    for item_id in keys(items)
        if item_id == itemid
            continue
        end
        otheritem = items[item_id]
        otherx, othery = otheritem.coords
        if direction == 1
       
            if otherx == itemx
                if otheritem.direction == 1 || # moving together
                    (otheritem.direction ==2 && abs(othery - IO[2]) <= 1) # moving to the front of this item makes no sense
                    continue
                else
                    return item_id, otherx, othery# great candidate 
                end
            elseif ((IO[1] < itemx && otherx > itemx) || (IO[1] > itemx && otherx < itemx)) && candid_aligned(othery, itemy) # depends on io
                xstart = min(itemx, otherx)
                xend = max(itemx, otherx)
                path_blocked = false
                for x in xstart:xend
                    if blockmat[x, itemy] == 1 || matrix[x, itemy] == item_id
                        path_blocked = true
                        break
                    end
                end
                if !path_blocked && 
                    ((IO[1] < itemx && otherx > itemx && otherx < nearestitemx) || # order: IO, item, other
                     (IO[1] > itemx && otherx < itemx && otherx > nearestitemx)) # order: other, item, IO
                    nearestitemid = item_id
                    nearestitemx = otherx
                end
            end
        elseif direction == 2
            if othery == itemy
                if otheritem.direction == 2 || # moving together
                    (otheritem.direction ==1 && abs(otherx - IO[1]) <= 1) ||  # moving to the front of this item makes no sense
                    (otherx - IO[1]) == 0  || # other item at depot
                    !(IO[1] < itemx && otherx > itemx) || (IO[1] > itemx && otherx < itemx) # this item doesn help other item
                    continue
                else
                    return item_id, otherx, othery # great candidate 
                end
            elseif othery > itemy && candid_aligned(otherx, itemx)
                ystart = min(itemy, othery)
                yend = max(itemy, othery)
                path_blocked = false
                for y in ystart:yend
                    if blockmat[itemx, y] == 1 || matrix[itemx, y] == item_id
                        path_blocked = true
                        break
                    end
                end
                if !path_blocked && othery < nearestitemy
                    nearestitemid = item_id
                    nearestitemy = othery
                end
            end
        end
    end
    return nearestitemid, nearestitemx, nearestitemy
end

function directserve_makespan!(iteration, matrix, items, escorts, escortid, urgcusts, blockmat, IO)
    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords

    # Get coordinates of escorts that have not moved this iteration and are not this escort
    other_escorts_coords = [(escorts[esc].coords[1], escorts[esc].coords[2]) for esc in keys(escorts) if esc != escortid]
   
    distx, disty = size(matrix, 1)+1, size(matrix, 2)+1
    closestx , closesty = 0 , 0 
    sortedkeys = sort_keys_by_distance(items, IO, true) # sort by distance to IO

    # Sort urgent customers by their urgency and distance to IO
   

    # CAN WE SERVE AN URGENT CUSTOMER DIRECTLY IN NEXT ITERATION? 
  
    # CAN WE SERVE ANOTHER CUSTOMER IN NEXT ITERATION? 
    for itemid in sortedkeys # try serve item in next iteration 
        itemx, itemy = items[itemid].coords
        if ((IO[1] < itemx && esc_x < itemx) ||  # check if we can move escort to item path on X
            (IO[1] > itemx && esc_x > itemx)) && itemx != esc_x
            ygap = abs(esc_y - itemy)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords # if there exists an escort ready to serve this item we dont block it
                if oy == itemy
                    if (IO[1] > itemx && esc_x > itemx) &&  # item going right we want to avoid itemx-ox-escx
                        (itemx < esc_x && ox < esc_x && itemx < ox) # esc_x < ox && itemx <ox || itemx < esc_x && ox < esc_x && itemx < ox # If going left, check if there's an escort further left
                        skipItem = true
                        break
                    elseif (IO[1] < itemx && esc_x < itemx) &&  # item goes left. we want to avoid escx-ox-itemx
                        (itemx > esc_x && ox > esc_x && itemx> ox )# esc_x > ox && itemx >ox || itemx> esc_x && ox > esc_x && itemx > ox  # If going right, check if there's an escort further right
                        skipItem = true
                        break
                    end
                end
            end
            if skipItem
                continue
            end
            if ygap <= disty && ygap > 0 # if gap is 0 we could have served, there must be a reason we didnt
                ymin = min(esc_y, itemy)
                ymax = max(esc_y, itemy)

               
                for y in ymin:ymax
                    if blockmat[esc_x, y] == 1 || haskey(items, matrix[esc_x, y])
                        path_blocked = true
                        break
                    end
                end
                if IO[1] > min(itemx, esc_x) && IO[1] < max(itemx, esc_x) # can serve but effects badly 
                    for x in min(esc_x, IO[1]):max(esc_x, IO[1])
                        if haskey(items, matrix[x, itemy]) 
                            path_blocked = true
                            break
                        end
                    end
                end
            else 
                continue
            end
            if !path_blocked || (ygap == 0 && ((esc_x < itemx && IO[1] < itemx) || (esc_x > itemx && IO[1] > itemx)))
                itemscoords = generatefuturecoords(items, escorts, 1, escortid, itemid, matrix, IO)
                sameycoords = filter(x -> x[2] == itemy && x[1] >= min(itemx, esc_x) && x[1] <= max(itemx, esc_x), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in sameycoords])
                if minDist  > length(keys(items)) || # item far out from IO
                    path_to_io_exists_if(matrix, itemscoords, IO)   # check with A* if this movement would cause some stupid block
                    disty = ygap 
                    closesty = itemid
                else# else we ban it for next iteration to simplify computation on assignment! 
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
        if esc_y < itemy # check if we can move escort to item path on Y 
            xgap = abs(esc_x - itemx)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords
                if ox == itemx && oy < itemy
                    skipItem = true
                    break
                end
            end
            if skipItem
                continue
            end
            if xgap <= distx && xgap > 0
                xmin = min(esc_x, itemx)
                xmax = max(esc_x, itemx)
                for x in xmin:xmax
                    if blockmat[x, esc_y] == 1 || haskey(items, matrix[x, esc_y])
                        path_blocked = true
                        break
                    end
                end
            else
                continue
            end
            if !path_blocked || xgap==0 # if gap is 0 we could have served, there must be a reason we didnt 
                itemscoords = generatefuturecoords(items, escorts,2, escortid, itemid, matrix, IO)
                samexcoords = filter(x -> x[1] == itemx && x[2] >= min(itemy, esc_y) && x[2] <= max(itemy, esc_y), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in samexcoords])
                if  minDist > length(keys(items)) ||
                    path_to_io_exists_if(matrix, itemscoords, IO) # check with A* if this movement would cause some stupid block
                    distx = xgap
                    closestx = itemid
                else 
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
    end
    # If we could serve an item we move to there
    if distx < disty  && closestx !=0 # go in front of item in Y direction
        if closestx in urgcusts 
            filter!(id -> id != closestx, urgcusts)
        end
        candx , candy = items[closestx].coords
        return true, (candx, esc_y)
    end
    if distx >= disty && closesty !=0 # go in path of item in X direction
        if closesty in urgcusts 
            filter!(id -> id != closesty, urgcusts)
        end
        candx , candy = items[closesty].coords
        return true, (esc_x, candy)
    end
    return false , (esc_x, esc_y)
end
function directserve_flow_r!(iteration, matrix, items, escorts, escortid, urgcusts, blockmat, IO)
    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords
    
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    # Get coordinates of escorts that have not moved this iteration and are not this escort
    other_escorts_coords = [(escorts[esc].coords[1], escorts[esc].coords[2]) for esc in keys(escorts) if esc != escortid]
   
    distx, disty = size(matrix, 1)+1, size(matrix, 2)+1
    closestx , closesty = 0 , 0 
    sortedkeys = sort_keys_by_distance(items, IO, true) # sort by distance to IO

    # Sort urgent customers by their urgency and distance to IO
    sorted_urgkeys = sort_urgkeys_by_distance_toescort_grasp(items, urgcusts,(esc_x, esc_y), true, 0.8)

    # CAN WE SERVE AN URGENT CUSTOMER DIRECTLY IN NEXT ITERATION? 
    for itemid in sorted_urgkeys # try serve item in next iteration 
        itemx, itemy = items[itemid].coords
        if ((IO[1] < itemx && esc_x < itemx) ||  # check if we can move escort to item path on X
            (IO[1] > itemx && esc_x > itemx)) && itemx != esc_x
            ygap = abs(esc_y - itemy)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords # if there exists an escort ready to serve this item we dont block it
                if oy == itemy
                    if (IO[1] > itemx && esc_x > itemx) &&  # item going right we want to avoid itemx-ox-escx
                        (itemx < esc_x && ox < esc_x && itemx < ox) # esc_x < ox && itemx <ox || itemx < esc_x && ox < esc_x && itemx < ox # If going left, check if there's an escort further left
                        skipItem = true
                        break
                    elseif (IO[1] < itemx && esc_x < itemx) &&  # item goes left. we want to avoid escx-ox-itemx
                        (itemx > esc_x && ox > esc_x && itemx> ox )# esc_x > ox && itemx >ox || itemx> esc_x && ox > esc_x && itemx > ox  # If going right, check if there's an escort further right
                        skipItem = true
                        break
                    end
                end
            end
            if skipItem
                continue
            end
            if ygap <= disty && ygap > 0 # if gap is 0 we could have served, there must be a reason we didnt
                ymin = min(esc_y, itemy)
                ymax = max(esc_y, itemy)

               
                for y in ymin:ymax
                    if blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys
                        path_blocked = true
                        break
                    end
                end
                if IO[1] > min(itemx, esc_x) && IO[1] < max(itemx, esc_x) # can serve but effects badly 
                    for x in min(esc_x, IO[1]):max(esc_x, IO[1])
                        if matrix[x, itemy] in allkeys 
                            path_blocked = true
                            break
                        end
                    end
                end
            else 
                continue
            end
            if !path_blocked || (ygap == 0 && ((esc_x < itemx && IO[1] < itemx) || (esc_x > itemx && IO[1] > itemx)))
                itemscoords = generatefuturecoords(items, escorts, 1, escortid, itemid, matrix, IO)
                sameycoords = filter(x -> x[2] == itemy && x[1] >= min(itemx, esc_x) && x[1] <= max(itemx, esc_x), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in sameycoords])
                if minDist  > length(keys(items)) || # item far out from IO
                    path_to_io_exists_if(matrix, itemscoords, IO)   # check with A* if this movement would cause some stupid block
                    disty = ygap 
                    closesty = itemid
                else# else we ban it for next iteration to simplify computation on assignment! 
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
        if esc_y < itemy # check if we can move escort to item path on Y 
            xgap = abs(esc_x - itemx)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords
                if ox == itemx && oy < itemy
                    skipItem = true
                    break
                end
            end
            if skipItem
                continue
            end
            if xgap <= distx && xgap > 0
                xmin = min(esc_x, itemx)
                xmax = max(esc_x, itemx)
                for x in xmin:xmax
                    if blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys
                        path_blocked = true
                        break
                    end
                end
            else
                continue
            end
            if !path_blocked || xgap==0 # if gap is 0 we could have served, there must be a reason we didnt 
                itemscoords = generatefuturecoords(items, escorts,2, escortid, itemid, matrix, IO)
                samexcoords = filter(x -> x[1] == itemx && x[2] >= min(itemy, esc_y) && x[2] <= max(itemy, esc_y), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in samexcoords])
                if  minDist > length(keys(items)) ||
                    path_to_io_exists_if(matrix, itemscoords, IO) # check with A* if this movement would cause some stupid block
                    distx = xgap
                    closestx = itemid
                else 
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
    end
    # If we could serve an item we move to there
    if distx < disty  && closestx !=0 # go in front of item in Y direction
        if closestx in urgcusts 
            filter!(id -> id != closestx, urgcusts)
        end
        candx , candy = items[closestx].coords
        return true, (candx, esc_y)
    end
    if distx >= disty && closesty !=0 # go in path of item in X direction
        if closesty in urgcusts 
            filter!(id -> id != closesty, urgcusts)
        end
        candx , candy = items[closesty].coords
        return true, (esc_x, candy)
    end
    # CAN WE SERVE ANOTHER CUSTOMER IN NEXT ITERATION? 
    for itemid in setdiff(sortedkeys, urgcusts) # try serve item in next iteration 
        itemx, itemy = items[itemid].coords
        if ((IO[1] < itemx && esc_x < itemx) ||  # check if we can move escort to item path on X
            (IO[1] > itemx && esc_x > itemx)) && itemx != esc_x
            ygap = abs(esc_y - itemy)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords # if there exists an escort ready to serve this item we dont block it
                if oy == itemy
                    if (IO[1] > itemx && esc_x > itemx) &&  # item going right we want to avoid itemx-ox-escx
                        (itemx < esc_x && ox < esc_x && itemx < ox) # esc_x < ox && itemx <ox || itemx < esc_x && ox < esc_x && itemx < ox # If going left, check if there's an escort further left
                        skipItem = true
                        break
                    elseif (IO[1] < itemx && esc_x < itemx) &&  # item goes left. we want to avoid escx-ox-itemx
                        (itemx > esc_x && ox > esc_x && itemx> ox )# esc_x > ox && itemx >ox || itemx> esc_x && ox > esc_x && itemx > ox  # If going right, check if there's an escort further right
                        skipItem = true
                        break
                    end
                end
            end
            if skipItem
                continue
            end
            if ygap <= disty && ygap > 0 # if gap is 0 we could have served, there must be a reason we didnt
                ymin = min(esc_y, itemy)
                ymax = max(esc_y, itemy)

               
                for y in ymin:ymax
                    if blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys
                        path_blocked = true
                        break
                    end
                end
                if IO[1] > min(itemx, esc_x) && IO[1] < max(itemx, esc_x) # can serve but effects badly 
                    for x in min(esc_x, IO[1]):max(esc_x, IO[1])
                        if matrix[x, itemy] in allkeys 
                            path_blocked = true
                            break
                        end
                    end
                end
            else 
                continue
            end
            if !path_blocked || (ygap == 0 && ((esc_x < itemx && IO[1] < itemx) || (esc_x > itemx && IO[1] > itemx)))
                itemscoords = generatefuturecoords(items, escorts, 1, escortid, itemid, matrix, IO)
                sameycoords = filter(x -> x[2] == itemy && x[1] >= min(itemx, esc_x) && x[1] <= max(itemx, esc_x), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in sameycoords])
                if minDist  > length(keys(items)) || # item far out from IO
                    path_to_io_exists_if(matrix, itemscoords, IO)   # check with A* if this movement would cause some stupid block
                    disty = ygap 
                    closesty = itemid
                else# else we ban it for next iteration to simplify computation on assignment! 
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
        if esc_y < itemy # check if we can move escort to item path on Y 
            xgap = abs(esc_x - itemx)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords
                if ox == itemx && oy < itemy
                    skipItem = true
                    break
                end
            end
            if skipItem
                continue
            end
            if xgap <= distx && xgap > 0
                xmin = min(esc_x, itemx)
                xmax = max(esc_x, itemx)
                for x in xmin:xmax
                    if blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys
                        path_blocked = true
                        break
                    end
                end
            else
                continue
            end
            if !path_blocked || xgap==0 # if gap is 0 we could have served, there must be a reason we didnt 
                itemscoords = generatefuturecoords(items, escorts,2, escortid, itemid, matrix, IO)
                samexcoords = filter(x -> x[1] == itemx && x[2] >= min(itemy, esc_y) && x[2] <= max(itemy, esc_y), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in samexcoords])
                if  minDist > length(keys(items)) ||
                    path_to_io_exists_if(matrix, itemscoords, IO) # check with A* if this movement would cause some stupid block
                    distx = xgap
                    closestx = itemid
                else 
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
    end
    # If we could serve an item we move to there
    if distx < disty  && closestx !=0 # go in front of item in Y direction
        if closestx in urgcusts 
            filter!(id -> id != closestx, urgcusts)
        end
        candx , candy = items[closestx].coords
        return true, (candx, esc_y)
    end
    if distx >= disty && closesty !=0 # go in path of item in X direction
        if closesty in urgcusts 
            filter!(id -> id != closesty, urgcusts)
        end
        candx , candy = items[closesty].coords
        return true, (esc_x, candy)
    end
    return false , (esc_x, esc_y)
end
function directserve_flow!(iteration, matrix, items, escorts, escortid, urgcusts, blockmat, IO)
    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords
    
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    # Get coordinates of escorts that have not moved this iteration and are not this escort
    other_escorts_coords = [(escorts[esc].coords[1], escorts[esc].coords[2]) for esc in keys(escorts) if esc != escortid]
   
    distx, disty = size(matrix, 1)+1, size(matrix, 2)+1
    closestx , closesty = 0 , 0 
    sortedkeys = sort_keys_by_distance(items, IO, true) # sort by distance to IO

    # Sort urgent customers by their urgency and distance to IO
    sorted_urgkeys = sort_urgkeys_by_distance_toescort(items, urgcusts,(esc_x, esc_y), true)

    # CAN WE SERVE AN URGENT CUSTOMER DIRECTLY IN NEXT ITERATION? 
    for itemid in sorted_urgkeys # try serve item in next iteration 
        itemx, itemy = items[itemid].coords
        DEBUG_MOVE_TRACE[] && println("    [directserve URGENT X-outer] t=$iteration esc=$escortid pos=($esc_x,$esc_y) item=$itemid pos=($itemx,$itemy) enter=$(((IO[1] < itemx && esc_x < itemx) || (IO[1] > itemx && esc_x > itemx)) && itemx != esc_x)")
        if ((IO[1] < itemx && esc_x < itemx) ||  # check if we can move escort to item path on X
            (IO[1] > itemx && esc_x > itemx)) && itemx != esc_x
            ygap = abs(esc_y - itemy)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords # if there exists an escort ready to serve this item we dont block it
                if oy == itemy
                    if (IO[1] > itemx && esc_x > itemx) &&  # item going right we want to avoid itemx-ox-escx
                        (itemx < esc_x && ox < esc_x && itemx < ox) # esc_x < ox && itemx <ox || itemx < esc_x && ox < esc_x && itemx < ox # If going left, check if there's an escort further left
                        skipItem = true
                        break
                    elseif (IO[1] < itemx && esc_x < itemx) &&  # item goes left. we want to avoid escx-ox-itemx
                        (itemx > esc_x && ox > esc_x && itemx> ox )# esc_x > ox && itemx >ox || itemx> esc_x && ox > esc_x && itemx > ox  # If going right, check if there's an escort further right
                        skipItem = true
                        break
                    end
                end
            end
            DEBUG_MOVE_TRACE[] && println("    [directserve URGENT X-skip] t=$iteration esc=$escortid item=$itemid skipItem=$skipItem other_escorts=$other_escorts_coords")
            if skipItem
                continue
            end
            # LM mode: ygap==0 means we're already aligned with this item's
            # row right now -- possibly because another escort's push just
            # moved the item onto our row this same iteration. Under BM this
            # is assumed to mean assignment already looked at us and passed,
            # so skip; under LM, assignment ran BEFORE this iteration's
            # movement using stale positions, so an escort that becomes
            # aligned mid-iteration never gets picked up until next
            # iteration's assignment -- unless it stays put. Let ygap==0 fall
            # through so this escort can signal "stay here" instead of
            # wandering off via freeroam!.
            if ygap <= disty && (ygap > 0 || UNIT_STEP[]) # if gap is 0 we could have served, there must be a reason we didnt
                ymin = min(esc_y, itemy)
                ymax = max(esc_y, itemy)

               
                for y in ymin:ymax
                    DEBUG_MOVE_TRACE[] && println("      [directserve URGENT X-cell] t=$iteration esc=$escortid y=$y blockmat=$(blockmat[esc_x, y]) matrix=$(repr(matrix[esc_x, y])) inallkeys=$(matrix[esc_x, y] in allkeys)")
                    if blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys
                        path_blocked = true
                        break
                    end
                end
                if IO[1] > min(itemx, esc_x) && IO[1] < max(itemx, esc_x) # can serve but effects badly 
                    for x in min(esc_x, IO[1]):max(esc_x, IO[1])
                        if matrix[x, itemy] in allkeys 
                            path_blocked = true
                            break
                        end
                    end
                end
            else 
                continue
            end
            DEBUG_MOVE_TRACE[] && println("    [directserve URGENT X-path] t=$iteration esc=$escortid item=$itemid ygap=$ygap path_blocked=$path_blocked")
            if !path_blocked || (ygap == 0 && ((esc_x < itemx && IO[1] < itemx) || (esc_x > itemx && IO[1] > itemx)))
                itemscoords = generatefuturecoords(items, escorts, 1, escortid, itemid, matrix, IO)
                sameycoords = filter(x -> x[2] == itemy && x[1] >= min(itemx, esc_x) && x[1] <= max(itemx, esc_x), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in sameycoords])
                if minDist  > length(keys(items)) || # item far out from IO
                    path_to_io_exists_if(matrix, itemscoords, IO)   # check with A* if this movement would cause some stupid block
                    disty = ygap 
                    closesty = itemid
                else# else we ban it for next iteration to simplify computation on assignment! 
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
        if esc_y < itemy # check if we can move escort to item path on Y 
            xgap = abs(esc_x - itemx)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords
                if ox == itemx && oy < itemy
                    skipItem = true
                    break
                end
            end
            if skipItem
                continue
            end
            if xgap <= distx && xgap > 0
                xmin = min(esc_x, itemx)
                xmax = max(esc_x, itemx)
                for x in xmin:xmax
                    if blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys
                        path_blocked = true
                        break
                    end
                end
            else
                continue
            end
            if !path_blocked || xgap==0 # if gap is 0 we could have served, there must be a reason we didnt 
                itemscoords = generatefuturecoords(items, escorts,2, escortid, itemid, matrix, IO)
                samexcoords = filter(x -> x[1] == itemx && x[2] >= min(itemy, esc_y) && x[2] <= max(itemy, esc_y), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in samexcoords])
                if  minDist > length(keys(items)) ||
                    path_to_io_exists_if(matrix, itemscoords, IO) # check with A* if this movement would cause some stupid block
                    distx = xgap
                    closestx = itemid
                else 
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
    end
    # If we could serve an item we move to there
    if distx < disty  && closestx !=0 # go in front of item in Y direction
        if closestx in urgcusts 
            filter!(id -> id != closestx, urgcusts)
        end
        candx , candy = items[closestx].coords
        return true, (candx, esc_y)
    end
    if distx >= disty && closesty !=0 # go in path of item in X direction
        if closesty in urgcusts 
            filter!(id -> id != closesty, urgcusts)
        end
        candx , candy = items[closesty].coords
        return true, (esc_x, candy)
    end
    # CAN WE SERVE ANOTHER CUSTOMER IN NEXT ITERATION? 
    for itemid in setdiff(sortedkeys, urgcusts) # try serve item in next iteration 
        itemx, itemy = items[itemid].coords
        DEBUG_MOVE_TRACE[] && println("    [directserve X-outer] t=$iteration esc=$escortid pos=($esc_x,$esc_y) item=$itemid pos=($itemx,$itemy) enter=$(((IO[1] < itemx && esc_x < itemx) || (IO[1] > itemx && esc_x > itemx)) && itemx != esc_x)")
        if ((IO[1] < itemx && esc_x < itemx) ||  # check if we can move escort to item path on X
            (IO[1] > itemx && esc_x > itemx)) && itemx != esc_x
            ygap = abs(esc_y - itemy)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords # if there exists an escort ready to serve this item we dont block it
                if oy == itemy
                    if (IO[1] > itemx && esc_x > itemx) &&  # item going right we want to avoid itemx-ox-escx
                        (itemx < esc_x && ox < esc_x && itemx < ox) # esc_x < ox && itemx <ox || itemx < esc_x && ox < esc_x && itemx < ox # If going left, check if there's an escort further left
                        skipItem = true
                        break
                    elseif (IO[1] < itemx && esc_x < itemx) &&  # item goes left. we want to avoid escx-ox-itemx
                        (itemx > esc_x && ox > esc_x && itemx> ox )# esc_x > ox && itemx >ox || itemx> esc_x && ox > esc_x && itemx > ox  # If going right, check if there's an escort further right
                        skipItem = true
                        break
                    end
                end
            end
            DEBUG_MOVE_TRACE[] && println("    [directserve X-skip] t=$iteration esc=$escortid item=$itemid skipItem=$skipItem other_escorts=$other_escorts_coords")
            if skipItem
                continue
            end
            # LM mode: ygap==0 means we're already aligned with this item's
            # row right now -- possibly because another escort's push just
            # moved the item onto our row this same iteration. Under BM this
            # is assumed to mean assignment already looked at us and passed,
            # so skip; under LM, assignment ran BEFORE this iteration's
            # movement using stale positions, so an escort that becomes
            # aligned mid-iteration never gets picked up until next
            # iteration's assignment -- unless it stays put. Let ygap==0 fall
            # through so this escort can signal "stay here" instead of
            # wandering off via freeroam!.
            if ygap <= disty && (ygap > 0 || UNIT_STEP[]) # if gap is 0 we could have served, there must be a reason we didnt
                ymin = min(esc_y, itemy)
                ymax = max(esc_y, itemy)

               
                for y in ymin:ymax
                    if blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys
                        path_blocked = true
                        break
                    end
                end
                if IO[1] > min(itemx, esc_x) && IO[1] < max(itemx, esc_x) # can serve but effects badly 
                    for x in min(esc_x, IO[1]):max(esc_x, IO[1])
                        if matrix[x, itemy] in allkeys 
                            path_blocked = true
                            break
                        end
                    end
                end
            else 
                continue
            end
            DEBUG_MOVE_TRACE[] && println("    [directserve X-path] t=$iteration esc=$escortid item=$itemid ygap=$ygap path_blocked=$path_blocked")
            if !path_blocked || (ygap == 0 && ((esc_x < itemx && IO[1] < itemx) || (esc_x > itemx && IO[1] > itemx)))
                itemscoords = generatefuturecoords(items, escorts, 1, escortid, itemid, matrix, IO)
                sameycoords = filter(x -> x[2] == itemy && x[1] >= min(itemx, esc_x) && x[1] <= max(itemx, esc_x), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in sameycoords])
                if minDist  > length(keys(items)) || # item far out from IO
                    path_to_io_exists_if(matrix, itemscoords, IO)   # check with A* if this movement would cause some stupid block
                    disty = ygap 
                DEBUG_MOVE_TRACE[] && println("    [directserve X-path ACCEPT] esc=$escortid item=$itemid ygap=$ygap minDist=$minDist thresh=$(length(keys(items)))")
                    closesty = itemid
                else# else we ban it for next iteration to simplify computation on assignment! 
                    DEBUG_MOVE_TRACE[] && println("    [directserve X-path BAN] esc=$escortid item=$itemid ygap=$ygap minDist=$minDist thresh=$(length(keys(items))) path_to_io=$(path_to_io_exists_if(matrix, itemscoords, IO))")
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
        if esc_y < itemy # check if we can move escort to item path on Y 
            xgap = abs(esc_x - itemx)
            path_blocked = false ; skipItem = false
            for (ox, oy) in other_escorts_coords
                if ox == itemx && oy < itemy
                    skipItem = true
                    break
                end
            end
            if skipItem
                continue
            end
            if xgap <= distx && xgap > 0
                xmin = min(esc_x, itemx)
                xmax = max(esc_x, itemx)
                for x in xmin:xmax
                    if blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys
                        path_blocked = true
                        break
                    end
                end
            else
                continue
            end
            if !path_blocked || xgap==0 # if gap is 0 we could have served, there must be a reason we didnt 
                itemscoords = generatefuturecoords(items, escorts,2, escortid, itemid, matrix, IO)
                samexcoords = filter(x -> x[1] == itemx && x[2] >= min(itemy, esc_y) && x[2] <= max(itemy, esc_y), itemscoords) # as moving this item might move another item closer to depot
                minDist = minimum([abs(IO[1] - coord[1]) + abs(IO[2] - coord[2]) for coord in samexcoords])
                if  minDist > length(keys(items)) ||
                    path_to_io_exists_if(matrix, itemscoords, IO) # check with A* if this movement would cause some stupid block
                    distx = xgap
                    closestx = itemid
                else 
                    if !haskey(thisescort.banset, iteration+1)
                        thisescort.banset[iteration+1] = [itemid]
                    else
                        push!(thisescort.banset[iteration+1],itemid)
                    end
                end
            end
        end
    end
    # If we could serve an item we move to there
    if distx < disty  && closestx !=0 # go in front of item in Y direction
        if closestx in urgcusts 
            filter!(id -> id != closestx, urgcusts)
        end
        candx , candy = items[closestx].coords
        return true, (candx, esc_y)
    end
    if distx >= disty && closesty !=0 # go in path of item in X direction
        if closesty in urgcusts 
            filter!(id -> id != closesty, urgcusts)
        end
        candx , candy = items[closesty].coords
        return true, (esc_x, candy)
    end
    return false , (esc_x, esc_y)
end
function directserve_flow_multi_io!(iteration, matrix, items, escorts, escortid,
                                     urgcusts, blockmat, item_to_ios, all_ios)
    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords

    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    other_escorts_coords = [(escorts[esc].coords[1], escorts[esc].coords[2]) for esc in keys(escorts) if esc != escortid]

    distx, disty = size(matrix, 1)+1, size(matrix, 2)+1
    closestx, closesty = 0, 0

    # Sort all items by distance to their nearest assigned IO
    sortedkeys = sort(collect(keys(items)), by = itemid -> begin
        assigned = get(item_to_ios, itemid, all_ios)
        minimum(io -> abs(io[1] - items[itemid].coords[1]) + abs(io[2] - items[itemid].coords[2]), assigned)
    end)

    sorted_urgkeys = sort_urgkeys_by_distance_toescort(items, urgcusts, (esc_x, esc_y), true)

    function try_serve_item!(itemid)
        itemx, itemy = items[itemid].coords
        assigned_ios = get(item_to_ios, itemid, all_ios)

        # ── X-DIRECTION: escort moves to item's y-row ──────────────────────
        # Need an IO where escort and IO are on the same side of the item —
        # that's the IO this item is heading toward, and the escort can intercept
        item_io_x = nothing
        for io in assigned_ios
            if ((io[1] < itemx && esc_x < itemx) || (io[1] > itemx && esc_x > itemx)) && itemx != esc_x
                item_io_x = io
                break
            end
        end

        if item_io_x !== nothing
            iox, ioy = item_io_x
            ygap = abs(esc_y - itemy)
            path_blocked = false; skipItem = false

            for (ox, oy) in other_escorts_coords
                if oy == itemy
                    if (iox > itemx && esc_x > itemx) &&
                        (itemx < esc_x && ox < esc_x && itemx < ox)
                        skipItem = true; break
                    elseif (iox < itemx && esc_x < itemx) &&
                        (itemx > esc_x && ox > esc_x && itemx > ox)
                        skipItem = true; break
                    end
                end
            end

            if !skipItem && ygap <= disty && ygap > 0
                for y in min(esc_y, itemy):max(esc_y, itemy)
                    if blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys
                        path_blocked = true; break
                    end
                end
                # IO sitting between escort and item on x-axis makes the serve harmful
                if iox > min(itemx, esc_x) && iox < max(itemx, esc_x)
                    for x in min(esc_x, iox):max(esc_x, iox)
                        if matrix[x, itemy] in allkeys
                            path_blocked = true; break
                        end
                    end
                end

                if !path_blocked || (ygap == 0 && ((esc_x < itemx && iox < itemx) || (esc_x > itemx && iox > itemx)))
                    itemscoords = generatefuturecoords(items, escorts, 1, escortid, itemid, matrix, item_io_x)
                    sameycoords = filter(x -> x[2] == itemy && x[1] >= min(itemx, esc_x) && x[1] <= max(itemx, esc_x), itemscoords)
                    minDist = minimum([abs(iox - coord[1]) + abs(ioy - coord[2]) for coord in sameycoords])
                    if minDist > length(keys(items)) ||
                        path_to_io_exists_if(matrix, itemscoords, item_io_x)
                        disty = ygap
                        closesty = itemid
                    else
                        if !haskey(thisescort.banset, iteration+1)
                            thisescort.banset[iteration+1] = [itemid]
                        else
                            push!(thisescort.banset[iteration+1], itemid)
                        end
                    end
                end
            end
        end

        # ── Y-DIRECTION: escort moves to item's x-column ───────────────────
        # Geometry here doesn't depend on which IO — escort just needs to be below item.
        # Use nearest assigned IO only for the path validation calls.
        if esc_y < itemy
            xgap = abs(esc_x - itemx)
            path_blocked = false; skipItem = false

            for (ox, oy) in other_escorts_coords
                if ox == itemx && oy < itemy
                    skipItem = true; break
                end
            end

            if !skipItem && xgap <= distx && xgap > 0
                for x in min(esc_x, itemx):max(esc_x, itemx)
                    if blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys
                        path_blocked = true; break
                    end
                end

                if !path_blocked || xgap == 0
                    item_io_y = argmin(io -> abs(io[1] - itemx) + abs(io[2] - itemy), assigned_ios)
                    iox, ioy = item_io_y
                    itemscoords = generatefuturecoords(items, escorts, 2, escortid, itemid, matrix, item_io_y)
                    samexcoords = filter(x -> x[1] == itemx && x[2] >= min(itemy, esc_y) && x[2] <= max(itemy, esc_y), itemscoords)
                    minDist = minimum([abs(iox - coord[1]) + abs(ioy - coord[2]) for coord in samexcoords])
                    if minDist > length(keys(items)) ||
                        path_to_io_exists_if(matrix, itemscoords, item_io_y)
                        distx = xgap
                        closestx = itemid
                    else
                        if !haskey(thisescort.banset, iteration+1)
                            thisescort.banset[iteration+1] = [itemid]
                        else
                            push!(thisescort.banset[iteration+1], itemid)
                        end
                    end
                end
            end
        end
    end

    # Urgent customers first
    for itemid in sorted_urgkeys
        ITEM_LOCK_ENABLED[] && !item_lock_ok(escortid, itemid) && continue
        try_serve_item!(itemid)
    end

    if distx < disty && closestx != 0
        if closestx in urgcusts; filter!(id -> id != closestx, urgcusts); end
        return true, (items[closestx].coords[1], esc_y)
    end
    if distx >= disty && closesty != 0
        if closesty in urgcusts; filter!(id -> id != closesty, urgcusts); end
        return true, (esc_x, items[closesty].coords[2])
    end

    # Then non-urgent
    for itemid in setdiff(sortedkeys, urgcusts)
        ITEM_LOCK_ENABLED[] && !item_lock_ok(escortid, itemid) && continue
        try_serve_item!(itemid)
    end

    if distx < disty && closestx != 0
        return true, (items[closestx].coords[1], esc_y)
    end
    if distx >= disty && closesty != 0
        return true, (esc_x, items[closesty].coords[2])
    end

    return false, (esc_x, esc_y)
end
"""
GRASP version of directserve_flow_multi_io! — identical except urgent items are
visited in a randomized (GRASP, α=0.8) order instead of strict distance order,
matching the same swap used in directserve_flow_r! for the single-IO case.
"""
function directserve_flow_multi_io_r!(iteration, matrix, items, escorts, escortid,
                                     urgcusts, blockmat, item_to_ios, all_ios)
    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords

    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    other_escorts_coords = [(escorts[esc].coords[1], escorts[esc].coords[2]) for esc in keys(escorts) if esc != escortid]

    distx, disty = size(matrix, 1)+1, size(matrix, 2)+1
    closestx, closesty = 0, 0

    # Sort all items by distance to their nearest assigned IO
    sortedkeys = sort(collect(keys(items)), by = itemid -> begin
        assigned = get(item_to_ios, itemid, all_ios)
        minimum(io -> abs(io[1] - items[itemid].coords[1]) + abs(io[2] - items[itemid].coords[2]), assigned)
    end)

    sorted_urgkeys = sort_urgkeys_by_distance_toescort_grasp(items, urgcusts, (esc_x, esc_y), true, 0.8)

    function try_serve_item!(itemid)
        itemx, itemy = items[itemid].coords
        assigned_ios = get(item_to_ios, itemid, all_ios)

        # ── X-DIRECTION: escort moves to item's y-row ──────────────────────
        # Need an IO where escort and IO are on the same side of the item —
        # that's the IO this item is heading toward, and the escort can intercept
        item_io_x = nothing
        for io in assigned_ios
            if ((io[1] < itemx && esc_x < itemx) || (io[1] > itemx && esc_x > itemx)) && itemx != esc_x
                item_io_x = io
                break
            end
        end

        if item_io_x !== nothing
            iox, ioy = item_io_x
            ygap = abs(esc_y - itemy)
            path_blocked = false; skipItem = false

            for (ox, oy) in other_escorts_coords
                if oy == itemy
                    if (iox > itemx && esc_x > itemx) &&
                        (itemx < esc_x && ox < esc_x && itemx < ox)
                        skipItem = true; break
                    elseif (iox < itemx && esc_x < itemx) &&
                        (itemx > esc_x && ox > esc_x && itemx > ox)
                        skipItem = true; break
                    end
                end
            end

            if !skipItem && ygap <= disty && ygap > 0
                for y in min(esc_y, itemy):max(esc_y, itemy)
                    if blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys
                        path_blocked = true; break
                    end
                end
                # IO sitting between escort and item on x-axis makes the serve harmful
                if iox > min(itemx, esc_x) && iox < max(itemx, esc_x)
                    for x in min(esc_x, iox):max(esc_x, iox)
                        if matrix[x, itemy] in allkeys
                            path_blocked = true; break
                        end
                    end
                end

                if !path_blocked || (ygap == 0 && ((esc_x < itemx && iox < itemx) || (esc_x > itemx && iox > itemx)))
                    itemscoords = generatefuturecoords(items, escorts, 1, escortid, itemid, matrix, item_io_x)
                    sameycoords = filter(x -> x[2] == itemy && x[1] >= min(itemx, esc_x) && x[1] <= max(itemx, esc_x), itemscoords)
                    minDist = minimum([abs(iox - coord[1]) + abs(ioy - coord[2]) for coord in sameycoords])
                    if minDist > length(keys(items)) ||
                        path_to_io_exists_if(matrix, itemscoords, item_io_x)
                        disty = ygap
                        closesty = itemid
                    else
                        if !haskey(thisescort.banset, iteration+1)
                            thisescort.banset[iteration+1] = [itemid]
                        else
                            push!(thisescort.banset[iteration+1], itemid)
                        end
                    end
                end
            end
        end

        # ── Y-DIRECTION: escort moves to item's x-column ───────────────────
        # Geometry here doesn't depend on which IO — escort just needs to be below item.
        # Use nearest assigned IO only for the path validation calls.
        if esc_y < itemy
            xgap = abs(esc_x - itemx)
            path_blocked = false; skipItem = false

            for (ox, oy) in other_escorts_coords
                if ox == itemx && oy < itemy
                    skipItem = true; break
                end
            end

            if !skipItem && xgap <= distx && xgap > 0
                for x in min(esc_x, itemx):max(esc_x, itemx)
                    if blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys
                        path_blocked = true; break
                    end
                end

                if !path_blocked || xgap == 0
                    item_io_y = argmin(io -> abs(io[1] - itemx) + abs(io[2] - itemy), assigned_ios)
                    iox, ioy = item_io_y
                    itemscoords = generatefuturecoords(items, escorts, 2, escortid, itemid, matrix, item_io_y)
                    samexcoords = filter(x -> x[1] == itemx && x[2] >= min(itemy, esc_y) && x[2] <= max(itemy, esc_y), itemscoords)
                    minDist = minimum([abs(iox - coord[1]) + abs(ioy - coord[2]) for coord in samexcoords])
                    if minDist > length(keys(items)) ||
                        path_to_io_exists_if(matrix, itemscoords, item_io_y)
                        distx = xgap
                        closestx = itemid
                    else
                        if !haskey(thisescort.banset, iteration+1)
                            thisescort.banset[iteration+1] = [itemid]
                        else
                            push!(thisescort.banset[iteration+1], itemid)
                        end
                    end
                end
            end
        end
    end

    # Urgent customers first
    for itemid in sorted_urgkeys
        ITEM_LOCK_ENABLED[] && !item_lock_ok(escortid, itemid) && continue
        try_serve_item!(itemid)
    end

    if distx < disty && closestx != 0
        if closestx in urgcusts; filter!(id -> id != closestx, urgcusts); end
        return true, (items[closestx].coords[1], esc_y)
    end
    if distx >= disty && closesty != 0
        if closesty in urgcusts; filter!(id -> id != closesty, urgcusts); end
        return true, (esc_x, items[closesty].coords[2])
    end

    # Then non-urgent
    for itemid in setdiff(sortedkeys, urgcusts)
        ITEM_LOCK_ENABLED[] && !item_lock_ok(escortid, itemid) && continue
        try_serve_item!(itemid)
    end

    if distx < disty && closestx != 0
        return true, (items[closestx].coords[1], esc_y)
    end
    if distx >= disty && closesty != 0
        return true, (esc_x, items[closesty].coords[2])
    end

    return false, (esc_x, esc_y)
end
function urgserve_r!(iteration, matrix, items, escorts, escortid, urgmats, IO)
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords
    
    # MUST WE SERVE A CUSTOMER SOON? 3-4 Steps 
    if !isempty(keys(urgmats))
        urgentassignmentdict = Dict{String, Tuple{Int, Int, Int}}() # itemid, steps, x, y
        for urgitem in shuffle(Random.default_rng(), collect(keys(urgmats)))
            urgmat= urgmats[urgitem]
            urgx,urgy = items[urgitem].coords
            skip_urgitem = false
            for escid in keys(escorts)
                if escid == escortid
                    continue
                end
                otherescx, otherescy = escorts[escid].coords
                if urgmat[otherescx, otherescy] == 2
                    skip_urgitem = true
                    break
                end
            end
            if skip_urgitem
                continue
            end                 
            foundy = false; foundx = false; completedy = false; completedx = false; distance = Inf; dir = urgx > IO[1] ? -1 : 1
            candidy = 0; xin = deepcopy(esc_x); yin = deepcopy(esc_y); candidx = 0; gapy = Inf ; gapx = Inf
            if urgmat[esc_x, esc_y] == 2
                #println("Escort is 2 steps away from urgent item, should have served in previous step")
                continue 
            end
            while !completedy 
                if ( esc_x < urgx && IO[1] < urgx) ||
                    (esc_x > urgx && IO[1] > urgx)
                    completedy = true
                    break
                end
                if  (esc_y>=urgy)
                    for y in esc_y-1:-1:1
                        if urgmat[xin, y] == 2
                            candidy = y
                            gapy = abs(esc_y-y )
                            foundy = true
                            break
                        elseif urgmat[xin, y] == 1
                            foundy = false
                        end
                    end
                    if foundy || ( xin + dir < 1 ||
                        xin + dir > size(matrix, 1) ||
                            (dir == -1 && xin + dir <= urgx) ||
                            (dir == 1 && xin + dir > urgx))
                        completedy = true  # we cannot move anymore, if we havent found a 2 we cannot move
                    else
                        xin += dir
                    end
                elseif (esc_y < urgy-1) 
                    for y in esc_y+1:urgy-1
                        if urgmat[xin, y] == 2
                            candidy = y
                            gapy = abs(esc_y-y )
                            foundy = true
                            break
                        elseif urgmat[xin, y] == 1
                            foundy = false
                        end
                    end
                    if foundy || ( xin + dir < 1 ||
                        xin + dir > size(matrix, 1) ||
                            (dir == -1 && xin + dir <= urgx) ||
                            (dir == 1 && xin + dir > urgx))
                        completedy = true  # we cannot move anymore, if we havent found a 2 we cannot move
                    else
                        xin += dir
                    end
        
                else
                    completedy = true
                end
                
            end
            while !completedx
                if esc_y<=urgy
                    completedx = true
                    break
                end
                if esc_x>=urgx
                    if dir == -1 # escort on the right side or urg: IO-urg-esc
                        for x in esc_x-1:-1:IO[1]
                            if urgmat[x, esc_y] == 2
                                candidx = x
                                gapx = abs(esc_x - x)
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                                break
                            end
                        end
                    else # escort on the right side of urg : urg-esc-IO or urg-IO-esc
                        for x in esc_x-1:-1:urgx
                            if urgmat[x, esc_y] == 2
                                candidx = x
                                gapx = abs(esc_x - x)
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                                break
                            end
                        end
                    end
                    if foundx || yin -1 <= urgy
                        completedx = true # we cannot move anymore, if we havent found a 2 we cannot move
                    else
                        yin -= 1
                    end
                elseif esc_x<urgx-1
                    if dir ==1# escort left of item, item left of IO
                        for x in esc_x+1:IO[1]
                            if urgmat[x, esc_y] == 2
                                candidx = x
                                gapx = abs(esc_x - x)
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                            end
                        end
                    else # escort left of item, item right of IO
                        for x in esc_x+1:urgx-1
                            if urgmat[x, esc_y] == 2
                                candidx = x
                                gapx = abs(esc_x - x)
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                            end
                        end
                        for x in esc_x-1:-1:1
                            if urgmat[x, esc_y] == 2
                                gapotherx = abs(esc_x - x)
                                if gapotherx < gapx
                                    gapx = gapotherx
                                    candidx = x
                                end                                
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                            end
                        end
                    end
                    if foundx || yin -1 <= urgy
                        completedx = true # we cannot move anymore, if we havent found a 2 we cannot move
                    else
                        yin -= 1
                    end
                else
                    completedx = true
                end
            end
            if foundy && foundx
                onx = xin == esc_x ? 1 : 0
                ony = yin == esc_y ? 1 : 0
                if onx + ony == 2 # 3 step both fine
                    if gapx < gapy
                        candidx = esc_x 
                    else 
                        candidy = esc_y
                    end
                elseif onx == 1  # 3 step go down with escort
                    candidy = esc_y; distance = gapx
                elseif ony == 1 # 3 step go left in with escort (if escort right out of item)
                    candidx = esc_x;  distance = gapy
                else # 4 step 
                    gapx = gapx + 5*(abs(esc_y - yin))
                    gapy = gapy + 5*(abs(esc_x- xin))
                    if gapx < gapy
                        candidy = yin; candidx = esc_x ; distance = gapx
                    else
                        candidx = xin ; candidy = esc_y ; distance = gapy
                    end
                end
            elseif foundy
                onx = xin == esc_x ? 1 : 0
                if onx ==1 
                    candidx = esc_x ; distance = gapy
                else # 4 Step
                    gapy = gapy + 5*(abs(esc_x- xin))
                    candidx = xin ; candidy = esc_y ; distance = gapy
                end
            elseif foundx 
                ony = yin == esc_y ? 1 : 0
                if ony ==1 
                    candidy = yin; distance = gapx
                else # 4 Step
                    gapx = gapx + 5*(abs(esc_y - yin))
                    candidy = yin; candidx = esc_x;  distance = gapx # go down 
                end
            end
            # we check how many steps to get to a 2 in the matrix from the position and how far
            if distance != Inf && candidx != 0 && candidy!=0 && !(haskey(escorts, matrix[candidx, candidy]))
                urgentassignmentdict[urgitem] =(distance , candidx,candidy) 
            end
        end

        if !isempty(urgentassignmentdict)
            min_distance = Inf
            min_key = ""
            min_tuple = ()
            for (key, value) in urgentassignmentdict
                if value[1] < min_distance
                    newx, newy = value[2], value[3]
                    skipthis = false
                    if esc_x == newx
                        ystart, yend = min(esc_y, newy), max(esc_y, newy)
                        for y in ystart:yend
                            if matrix[esc_x, y] in allkeys
                                skipthis = true
                                break
                            end
                        end
                    elseif esc_y == newy
                        xstart, xend = min(esc_x, newx), max(esc_x, newx)
                        for x in xstart:xend
                            if matrix[x, esc_y] in allkeys
                                skipthis = true
                                break
                            end
                        end
                    end
                    if skipthis
                        continue
                    elseif !((newx, newy) in escorts[escortid].tabu)
                        min_distance = value[1]
                        min_key = key
                        min_tuple = value
                    end
                end
            end 
            
            if  !isempty(min_tuple)
                delete!(urgmats, min_key)
                otheritems = setdiff(keys(urgentassignmentdict), [min_key])
                if !haskey(thisescort.banset, iteration+1)
                    thisescort.banset[iteration+1] = Vector{String}(collect(otheritems))
                else
                    append!(thisescort.banset[iteration+1], otheritems)
                end
                #println("$iteration :$escortid-> $min_key movement decided by urgency policy")
                return true, (min_tuple[2], min_tuple[3]) # we also need some sort of commitment. using the banset I assume TODO 
            end
        end 
    end
   
    return false, (esc_x, esc_y ) # cannot move
end
function urgserve!(iteration, matrix, items, escorts, escortid, urgmats, IO)
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords
    
    # MUST WE SERVE A CUSTOMER SOON? 3-4 Steps 
    if !isempty(keys(urgmats))
        urgentassignmentdict = Dict{String, Tuple{Int, Int, Int}}() # itemid, steps, x, y
        for urgitem in keys(urgmats)
            urgmat= urgmats[urgitem]
            urgx,urgy = items[urgitem].coords
            skip_urgitem = false
            for escid in keys(escorts)
                if escid == escortid
                    continue
                end
                otherescx, otherescy = escorts[escid].coords
                if urgmat[otherescx, otherescy] == 2
                    skip_urgitem = true
                    break
                end
            end
            if skip_urgitem
                continue
            end                 
            foundy = false; foundx = false; completedy = false; completedx = false; distance = Inf; dir = urgx > IO[1] ? -1 : 1
            candidy = 0; xin = deepcopy(esc_x); yin = deepcopy(esc_y); candidx = 0; gapy = Inf ; gapx = Inf
            if urgmat[esc_x, esc_y] == 2
                #println("Escort is 2 steps away from urgent item, should have served in previous step")
                continue 
            end
            while !completedy 
                if ( esc_x < urgx && IO[1] < urgx) ||
                    (esc_x > urgx && IO[1] > urgx)
                    completedy = true
                    break
                end
                if  (esc_y>=urgy)
                    for y in esc_y-1:-1:1
                        if urgmat[xin, y] == 2
                            candidy = y
                            gapy = abs(esc_y-y )
                            foundy = true
                            break
                        elseif urgmat[xin, y] == 1
                            foundy = false
                        end
                    end
                    if foundy || ( xin + dir < 1 ||
                        xin + dir > size(matrix, 1) ||
                            (dir == -1 && xin + dir <= urgx) ||
                            (dir == 1 && xin + dir > urgx))
                        completedy = true  # we cannot move anymore, if we havent found a 2 we cannot move
                    else
                        xin += dir
                    end
                elseif (esc_y < urgy-1) 
                    for y in esc_y+1:urgy-1
                        if urgmat[xin, y] == 2
                            candidy = y
                            gapy = abs(esc_y-y )
                            foundy = true
                            break
                        elseif urgmat[xin, y] == 1
                            foundy = false
                        end
                    end
                    if foundy || ( xin + dir < 1 ||
                        xin + dir > size(matrix, 1) ||
                            (dir == -1 && xin + dir <= urgx) ||
                            (dir == 1 && xin + dir > urgx))
                        completedy = true  # we cannot move anymore, if we havent found a 2 we cannot move
                    else
                        xin += dir
                    end
        
                else
                    completedy = true
                end
                
            end
            while !completedx
                if esc_y<=urgy
                    completedx = true
                    break
                end
                if esc_x>=urgx
                    if dir == -1 # escort on the right side or urg: IO-urg-esc
                        for x in esc_x-1:-1:IO[1]
                            if urgmat[x, esc_y] == 2
                                candidx = x
                                gapx = abs(esc_x - x)
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                                break
                            end
                        end
                    else # escort on the right side of urg : urg-esc-IO or urg-IO-esc
                        for x in esc_x-1:-1:urgx
                            if urgmat[x, esc_y] == 2
                                candidx = x
                                gapx = abs(esc_x - x)
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                                break
                            end
                        end
                    end
                    if foundx || yin -1 <= urgy
                        completedx = true # we cannot move anymore, if we havent found a 2 we cannot move
                    else
                        yin -= 1
                    end
                elseif esc_x<urgx-1
                    if dir ==1# escort left of item, item left of IO
                        for x in esc_x+1:IO[1]
                            if urgmat[x, esc_y] == 2
                                candidx = x
                                gapx = abs(esc_x - x)
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                            end
                        end
                    else # escort left of item, item right of IO
                        for x in esc_x+1:urgx-1
                            if urgmat[x, esc_y] == 2
                                candidx = x
                                gapx = abs(esc_x - x)
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                            end
                        end
                        for x in esc_x-1:-1:1
                            if urgmat[x, esc_y] == 2
                                gapotherx = abs(esc_x - x)
                                if gapotherx < gapx
                                    gapx = gapotherx
                                    candidx = x
                                end                                
                                foundx = true
                                break
                            elseif urgmat[x, esc_y] == 1
                                foundx = false
                            end
                        end
                    end
                    if foundx || yin -1 <= urgy
                        completedx = true # we cannot move anymore, if we havent found a 2 we cannot move
                    else
                        yin -= 1
                    end
                else
                    completedx = true
                end
            end
            if foundy && foundx
                onx = xin == esc_x ? 1 : 0
                ony = yin == esc_y ? 1 : 0
                if onx + ony == 2 # 3 step both fine
                    if gapx < gapy
                        candidx = esc_x 
                    else 
                        candidy = esc_y
                    end
                elseif onx == 1  # 3 step go down with escort
                    candidy = esc_y; distance = gapx
                elseif ony == 1 # 3 step go left in with escort (if escort right out of item)
                    candidx = esc_x;  distance = gapy
                else # 4 step 
                    gapx = gapx + 5*(abs(esc_y - yin))
                    gapy = gapy + 5*(abs(esc_x- xin))
                    if gapx < gapy
                        candidy = yin; candidx = esc_x ; distance = gapx
                    else
                        candidx = xin ; candidy = esc_y ; distance = gapy
                    end
                end
            elseif foundy
                onx = xin == esc_x ? 1 : 0
                if onx ==1 
                    candidx = esc_x ; distance = gapy
                else # 4 Step
                    gapy = gapy + 5*(abs(esc_x- xin))
                    candidx = xin ; candidy = esc_y ; distance = gapy
                end
            elseif foundx 
                ony = yin == esc_y ? 1 : 0
                if ony ==1 
                    candidy = yin; distance = gapx
                else # 4 Step
                    gapx = gapx + 5*(abs(esc_y - yin))
                    candidy = yin; candidx = esc_x;  distance = gapx # go down 
                end
            end
            # we check how many steps to get to a 2 in the matrix from the position and how far
            if distance != Inf && candidx != 0 && candidy!=0 && !(haskey(escorts, matrix[candidx, candidy]))
                urgentassignmentdict[urgitem] =(distance , candidx,candidy) 
            end
        end

        if !isempty(urgentassignmentdict)
            min_distance = Inf
            min_key = ""
            min_tuple = ()
            for (key, value) in urgentassignmentdict
                if value[1] < min_distance
                    newx, newy = value[2], value[3]
                    skipthis = false
                    if esc_x == newx
                        ystart, yend = min(esc_y, newy), max(esc_y, newy)
                        for y in ystart:yend
                            if matrix[esc_x, y] in allkeys
                                skipthis = true
                                break
                            end
                        end
                    elseif esc_y == newy
                        xstart, xend = min(esc_x, newx), max(esc_x, newx)
                        for x in xstart:xend
                            if matrix[x, esc_y] in allkeys
                                skipthis = true
                                break
                            end
                        end
                    end
                    if skipthis
                        continue
                    elseif !((newx, newy) in escorts[escortid].tabu)
                        min_distance = value[1]
                        min_key = key
                        min_tuple = value
                    end
                end
            end 
            
            if  !isempty(min_tuple)
                delete!(urgmats, min_key)
                otheritems = setdiff(keys(urgentassignmentdict), [min_key])
                if !haskey(thisescort.banset, iteration+1)
                    thisescort.banset[iteration+1] = Vector{String}(collect(otheritems))
                else
                    append!(thisescort.banset[iteration+1], otheritems)
                end
                #println("$iteration :$escortid-> $min_key movement decided by urgency policy")
                return true, (min_tuple[2], min_tuple[3]) # we also need some sort of commitment. using the banset I assume TODO 
            end
        end 
    end
   
    return false, (esc_x, esc_y ) # cannot move
end
function urgserve_multi_io!(iteration, matrix, items, escorts, escortid, urgmats, item_to_ios, all_ios)
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords

    # LM-only: a urgmat==1 cell that is just another escort parked more than two
    # cells away along the scan axis cannot obstruct this escort's single-cell
    # step, so the urgent-serve scans below treat it as passable instead of a
    # hard block (prevents the "sidestep around a far escort" walk). BM untouched.
    far_escort_block(cx, cy, axisdist) = UNIT_STEP[] && axisdist > 2 && haskey(escorts, matrix[cx, cy])

    # LM multi-IO: an escort may only urgent-serve a load whose target IO is its
    # own zone IO or an immediately-adjacent IO (in x). Loads two or more IOs
    # away are off-limits — stops an idle escort near IO1 being dragged the
    # width of the grid toward a far-corner urgent load just because nothing
    # nearer scored a candidate that iteration.
    lm_zone_gate = UNIT_STEP[] && length(all_ios) > 1
    ios_by_x = lm_zone_gate ? sort(collect(all_ios), by = io -> io[1]) : Tuple{Int,Int}[]
    esc_io_idx = lm_zone_gate ? argmin(i -> abs(ios_by_x[i][1] - esc_x), eachindex(ios_by_x)) : 0

    if !isempty(keys(urgmats))
        urgentassignmentdict = Dict{String, Tuple{Int, Int, Int}}()
        for urgitem in keys(urgmats)
            ITEM_LOCK_ENABLED[] && !item_lock_ok(escortid, urgitem) && continue
            urgmat = urgmats[urgitem]
            urgx, urgy = items[urgitem].coords

            # Look up this item's assigned IO — same choice as urgmats_multi_io used
            assigned = get(item_to_ios, urgitem, all_ios)
            item_io = argmin(io -> abs(io[1] - urgx), assigned)
            iox, ioy = item_io   # replaces IO[1]/IO[2] everywhere below

            if lm_zone_gate
                item_io_idx = argmin(i -> abs(ios_by_x[i][1] - iox), eachindex(ios_by_x))
                if abs(item_io_idx - esc_io_idx) > LM_URG_IO_RADIUS[]
                    continue
                end
            end

            skip_urgitem = false
            for escid in keys(escorts)
                if escid == escortid; continue; end
                otherescx, otherescy = escorts[escid].coords
                if urgmat[otherescx, otherescy] == 2
                    skip_urgitem = true; break
                end
            end
            if skip_urgitem; continue; end

            foundy = false; foundx = false; completedy = false; completedx = false
            distance = Inf
            dir = urgx > iox ? -1 : 1    # was: urgx > IO[1]
            candidy = 0; xin = deepcopy(esc_x); yin = deepcopy(esc_y)
            candidx = 0; gapy = Inf; gapx = Inf

            if urgmat[esc_x, esc_y] == 2
                continue
            end

            while !completedy
                if (esc_x < urgx && iox < urgx) ||    # was IO[1] < urgx
                    (esc_x > urgx && iox > urgx)       # was IO[1] > urgx
                    completedy = true; break
                end
                if esc_y >= urgy
                    for y in esc_y-1:-1:1
                        if urgmat[xin, y] == 2
                            candidy = y; gapy = abs(esc_y - y); foundy = true; break
                        elseif urgmat[xin, y] == 1 && !far_escort_block(xin, y, abs(esc_y - y))
                            foundy = false
                        end
                    end
                    if foundy || (xin + dir < 1 || xin + dir > size(matrix, 1) ||
                        (dir == -1 && xin + dir <= urgx) || (dir == 1 && xin + dir > urgx))
                        completedy = true
                    else
                        xin += dir
                    end
                elseif esc_y < urgy - 1
                    for y in esc_y+1:urgy-1
                        if urgmat[xin, y] == 2
                            candidy = y; gapy = abs(esc_y - y); foundy = true; break
                        elseif urgmat[xin, y] == 1 && !far_escort_block(xin, y, abs(esc_y - y))
                            foundy = false
                        end
                    end
                    if foundy || (xin + dir < 1 || xin + dir > size(matrix, 1) ||
                        (dir == -1 && xin + dir <= urgx) || (dir == 1 && xin + dir > urgx))
                        completedy = true
                    else
                        xin += dir
                    end
                else
                    completedy = true
                end
            end

            while !completedx
                if esc_y <= urgy; completedx = true; break; end
                if esc_x >= urgx
                    if dir == -1  # IO-urg-esc: search left toward IO
                        for x in esc_x-1:-1:iox    # was IO[1]
                            if urgmat[x, esc_y] == 2
                                candidx = x; gapx = abs(esc_x - x); foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false; break
                            end
                        end
                    else  # urg-esc-IO or urg-IO-esc: search left toward item
                        for x in esc_x-1:-1:urgx
                            if urgmat[x, esc_y] == 2
                                candidx = x; gapx = abs(esc_x - x); foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false; break
                            end
                        end
                    end
                    if foundx || yin - 1 <= urgy; completedx = true; else; yin -= 1; end
                elseif esc_x < urgx - 1
                    if dir == 1  # escort left of item, item left of IO: search right toward IO
                        for x in esc_x+1:iox    # was IO[1]
                            if urgmat[x, esc_y] == 2
                                candidx = x; gapx = abs(esc_x - x); foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false
                            end
                        end
                    else  # escort left of item, item right of IO
                        for x in esc_x+1:urgx-1
                            if urgmat[x, esc_y] == 2
                                candidx = x; gapx = abs(esc_x - x); foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false
                            end
                        end
                        for x in esc_x-1:-1:1
                            if urgmat[x, esc_y] == 2
                                gapotherx = abs(esc_x - x)
                                if gapotherx < gapx; gapx = gapotherx; candidx = x; end
                                foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false
                            end
                        end
                    end
                    if foundx || yin - 1 <= urgy; completedx = true; else; yin -= 1; end
                else
                    completedx = true
                end
            end

            # Distance scoring and candidate selection — unchanged from urgserve!
            if foundy && foundx
                onx = xin == esc_x ? 1 : 0; ony = yin == esc_y ? 1 : 0
                if onx + ony == 2
                    if gapx < gapy; candidx = esc_x; else; candidy = esc_y; end
                elseif onx == 1
                    candidy = esc_y; distance = gapx
                elseif ony == 1
                    candidx = esc_x; distance = gapy
                else
                    gapx = gapx + 5*(abs(esc_y - yin)); gapy = gapy + 5*(abs(esc_x - xin))
                    if gapx < gapy; candidy = yin; candidx = esc_x; distance = gapx
                    else; candidx = xin; candidy = esc_y; distance = gapy; end
                end
            elseif foundy
                onx = xin == esc_x ? 1 : 0
                if onx == 1; candidx = esc_x; distance = gapy
                else; gapy = gapy + 5*(abs(esc_x - xin)); candidx = xin; candidy = esc_y; distance = gapy; end
            elseif foundx
                ony = yin == esc_y ? 1 : 0
                if ony == 1; candidy = yin; distance = gapx
                else; gapx = gapx + 5*(abs(esc_y - yin)); candidy = yin; candidx = esc_x; distance = gapx; end
            end
            if distance != Inf && candidx != 0 && candidy != 0 && !(haskey(escorts, matrix[candidx, candidy]))
                urgentassignmentdict[urgitem] = (distance, candidx, candidy)
            end
            TRACE_ESC[] == escortid && println("    t=$iteration URG $escortid esc=($esc_x,$esc_y) urgitem=$urgitem urgpos=($urgx,$urgy) io=($iox,$ioy) foundx=$foundx foundy=$foundy -> cand=($candidx,$candidy) dist=$distance")
        end

        if !isempty(urgentassignmentdict)
            min_distance = Inf; min_key = ""; min_tuple = ()
            for (key, value) in urgentassignmentdict
                if value[1] < min_distance
                    newx, newy = value[2], value[3]; skipthis = false
                    if esc_x == newx
                        for y in min(esc_y, newy):max(esc_y, newy)
                            if matrix[esc_x, y] in allkeys; skipthis = true; break; end
                        end
                    elseif esc_y == newy
                        for x in min(esc_x, newx):max(esc_x, newx)
                            if matrix[x, esc_y] in allkeys; skipthis = true; break; end
                        end
                    end
                    if skipthis; continue
                    elseif !((newx, newy) in escorts[escortid].tabu)
                        min_distance = value[1]; min_key = key; min_tuple = value
                    end
                end
            end
            if !isempty(min_tuple)
                TRACE_ESC[] == escortid && println("    t=$iteration URG-WIN $escortid item=$min_key target=($(min_tuple[2]),$(min_tuple[3])) dist=$(min_tuple[1])  all=$urgentassignmentdict")
                delete!(urgmats, min_key)
                otheritems = setdiff(keys(urgentassignmentdict), [min_key])
                if !haskey(thisescort.banset, iteration+1)
                    thisescort.banset[iteration+1] = Vector{String}(collect(otheritems))
                else
                    append!(thisescort.banset[iteration+1], otheritems)
                end
                return true, (min_tuple[2], min_tuple[3])
            end
        end
    end
    return false, (esc_x, esc_y)
end
"""
GRASP version of urgserve_multi_io! — identical except urgent items are visited
in a randomized (shuffled) order instead of the arbitrary Dict key order, matching
the same swap used in urgserve_r! for the single-IO case.
"""
function urgserve_multi_io_r!(iteration, matrix, items, escorts, escortid, urgmats, item_to_ios, all_ios)
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords

    # LM-only: ignore a urgmat==1 cell that is just another escort parked more
    # than two cells away along the scan axis (see urgserve_multi_io!).
    far_escort_block(cx, cy, axisdist) = UNIT_STEP[] && axisdist > 2 && haskey(escorts, matrix[cx, cy])

    if !isempty(keys(urgmats))
        urgentassignmentdict = Dict{String, Tuple{Int, Int, Int}}()
        for urgitem in shuffle(Random.default_rng(), collect(keys(urgmats)))
            urgmat = urgmats[urgitem]
            urgx, urgy = items[urgitem].coords

            # Look up this item's assigned IO — same choice as urgmats_multi_io used
            assigned = get(item_to_ios, urgitem, all_ios)
            item_io = argmin(io -> abs(io[1] - urgx), assigned)
            iox, ioy = item_io   # replaces IO[1]/IO[2] everywhere below

            skip_urgitem = false
            for escid in keys(escorts)
                if escid == escortid; continue; end
                otherescx, otherescy = escorts[escid].coords
                if urgmat[otherescx, otherescy] == 2
                    skip_urgitem = true; break
                end
            end
            if skip_urgitem; continue; end

            foundy = false; foundx = false; completedy = false; completedx = false
            distance = Inf
            dir = urgx > iox ? -1 : 1    # was: urgx > IO[1]
            candidy = 0; xin = deepcopy(esc_x); yin = deepcopy(esc_y)
            candidx = 0; gapy = Inf; gapx = Inf

            if urgmat[esc_x, esc_y] == 2
                continue
            end

            while !completedy
                if (esc_x < urgx && iox < urgx) ||    # was IO[1] < urgx
                    (esc_x > urgx && iox > urgx)       # was IO[1] > urgx
                    completedy = true; break
                end
                if esc_y >= urgy
                    for y in esc_y-1:-1:1
                        if urgmat[xin, y] == 2
                            candidy = y; gapy = abs(esc_y - y); foundy = true; break
                        elseif urgmat[xin, y] == 1 && !far_escort_block(xin, y, abs(esc_y - y))
                            foundy = false
                        end
                    end
                    if foundy || (xin + dir < 1 || xin + dir > size(matrix, 1) ||
                        (dir == -1 && xin + dir <= urgx) || (dir == 1 && xin + dir > urgx))
                        completedy = true
                    else
                        xin += dir
                    end
                elseif esc_y < urgy - 1
                    for y in esc_y+1:urgy-1
                        if urgmat[xin, y] == 2
                            candidy = y; gapy = abs(esc_y - y); foundy = true; break
                        elseif urgmat[xin, y] == 1 && !far_escort_block(xin, y, abs(esc_y - y))
                            foundy = false
                        end
                    end
                    if foundy || (xin + dir < 1 || xin + dir > size(matrix, 1) ||
                        (dir == -1 && xin + dir <= urgx) || (dir == 1 && xin + dir > urgx))
                        completedy = true
                    else
                        xin += dir
                    end
                else
                    completedy = true
                end
            end

            while !completedx
                if esc_y <= urgy; completedx = true; break; end
                if esc_x >= urgx
                    if dir == -1  # IO-urg-esc: search left toward IO
                        for x in esc_x-1:-1:iox    # was IO[1]
                            if urgmat[x, esc_y] == 2
                                candidx = x; gapx = abs(esc_x - x); foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false; break
                            end
                        end
                    else  # urg-esc-IO or urg-IO-esc: search left toward item
                        for x in esc_x-1:-1:urgx
                            if urgmat[x, esc_y] == 2
                                candidx = x; gapx = abs(esc_x - x); foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false; break
                            end
                        end
                    end
                    if foundx || yin - 1 <= urgy; completedx = true; else; yin -= 1; end
                elseif esc_x < urgx - 1
                    if dir == 1  # escort left of item, item left of IO: search right toward IO
                        for x in esc_x+1:iox    # was IO[1]
                            if urgmat[x, esc_y] == 2
                                candidx = x; gapx = abs(esc_x - x); foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false
                            end
                        end
                    else  # escort left of item, item right of IO
                        for x in esc_x+1:urgx-1
                            if urgmat[x, esc_y] == 2
                                candidx = x; gapx = abs(esc_x - x); foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false
                            end
                        end
                        for x in esc_x-1:-1:1
                            if urgmat[x, esc_y] == 2
                                gapotherx = abs(esc_x - x)
                                if gapotherx < gapx; gapx = gapotherx; candidx = x; end
                                foundx = true; break
                            elseif urgmat[x, esc_y] == 1 && !far_escort_block(x, esc_y, abs(esc_x - x))
                                foundx = false
                            end
                        end
                    end
                    if foundx || yin - 1 <= urgy; completedx = true; else; yin -= 1; end
                else
                    completedx = true
                end
            end

            # Distance scoring and candidate selection — unchanged from urgserve!
            if foundy && foundx
                onx = xin == esc_x ? 1 : 0; ony = yin == esc_y ? 1 : 0
                if onx + ony == 2
                    if gapx < gapy; candidx = esc_x; else; candidy = esc_y; end
                elseif onx == 1
                    candidy = esc_y; distance = gapx
                elseif ony == 1
                    candidx = esc_x; distance = gapy
                else
                    gapx = gapx + 5*(abs(esc_y - yin)); gapy = gapy + 5*(abs(esc_x - xin))
                    if gapx < gapy; candidy = yin; candidx = esc_x; distance = gapx
                    else; candidx = xin; candidy = esc_y; distance = gapy; end
                end
            elseif foundy
                onx = xin == esc_x ? 1 : 0
                if onx == 1; candidx = esc_x; distance = gapy
                else; gapy = gapy + 5*(abs(esc_x - xin)); candidx = xin; candidy = esc_y; distance = gapy; end
            elseif foundx
                ony = yin == esc_y ? 1 : 0
                if ony == 1; candidy = yin; distance = gapx
                else; gapx = gapx + 5*(abs(esc_y - yin)); candidy = yin; candidx = esc_x; distance = gapx; end
            end
            if distance != Inf && candidx != 0 && candidy != 0 && !(haskey(escorts, matrix[candidx, candidy]))
                urgentassignmentdict[urgitem] = (distance, candidx, candidy)
            end
            TRACE_ESC[] == escortid && println("    t=$iteration URG $escortid esc=($esc_x,$esc_y) urgitem=$urgitem urgpos=($urgx,$urgy) io=($iox,$ioy) foundx=$foundx foundy=$foundy -> cand=($candidx,$candidy) dist=$distance")
        end

        if !isempty(urgentassignmentdict)
            min_distance = Inf; min_key = ""; min_tuple = ()
            for (key, value) in urgentassignmentdict
                if value[1] < min_distance
                    newx, newy = value[2], value[3]; skipthis = false
                    if esc_x == newx
                        for y in min(esc_y, newy):max(esc_y, newy)
                            if matrix[esc_x, y] in allkeys; skipthis = true; break; end
                        end
                    elseif esc_y == newy
                        for x in min(esc_x, newx):max(esc_x, newx)
                            if matrix[x, esc_y] in allkeys; skipthis = true; break; end
                        end
                    end
                    if skipthis; continue
                    elseif !((newx, newy) in escorts[escortid].tabu)
                        min_distance = value[1]; min_key = key; min_tuple = value
                    end
                end
            end
            if !isempty(min_tuple)
                delete!(urgmats, min_key)
                otheritems = setdiff(keys(urgentassignmentdict), [min_key])
                if !haskey(thisescort.banset, iteration+1)
                    thisescort.banset[iteration+1] = Vector{String}(collect(otheritems))
                else
                    append!(thisescort.banset[iteration+1], otheritems)
                end
                return true, (min_tuple[2], min_tuple[3])
            end
        end
    end
    return false, (esc_x, esc_y)
end
# Returns true if (x,y) is orthogonally adjacent to any escort other than escortid.
function is_escort_adjacent(x, y, escorts, escortid, blockmat)
    for (id, esc) in escorts
        id == escortid && continue
        ex, ey = esc.coords
        # Only a problem if the adjacent escort hasn't moved yet (blockmat still 0).
        # blockmat[ex,ey]==1 means it already moved this iteration and will vacate.
        if (abs(ex - x) + abs(ey - y)) == 1 && blockmat[ex, ey] == 0
            return true
        end
    end
    return false
end

# Given a target reached by moving in direction (dx, dy) from (orig_x, orig_y),
# step back one cell at a time until we find a non-adjacent position or run out of
# room.  Returns the best available position (may still be adjacent if no better
# option exists — movement is still preferable to staying).
function nudge_from_escorts(orig_x, orig_y, target_x, target_y, escorts, escortid, blockmat, matrix)
    if !is_escort_adjacent(target_x, target_y, escorts, escortid, blockmat)
        return (target_x, target_y)   # already fine
    end
    # Step back one cell towards the origin and check
    dx = target_x == orig_x ? 0 : (target_x > orig_x ? -1 : 1)
    dy = target_y == orig_y ? 0 : (target_y > orig_y ? -1 : 1)
    cx, cy = target_x + dx, target_y + dy
    if (cx != orig_x || cy != orig_y) &&                           # don't land back on start
       blockmat[cx, cy] == 0 &&                                     # not a blocked cell
       !(haskey(escorts, matrix[cx, cy])) &&                        # no other escort there
       !is_escort_adjacent(cx, cy, escorts, escortid, blockmat)     # not adjacent to unmoved escort
        return (cx, cy)
    end
    return (target_x, target_y)  # fall back to original target
end

# When the target IO sits on the top edge (last column), an escort already on that
# column that is blocked from advancing toward the IO in X cannot sidestep "up"
# (esc_y+1 is off-grid). The perpendicular escape is DOWNWARD instead. Mirror of the
# diresc==2 block in checkasternmat, scanning -y: returns the reachable y < esc_y with
# the lowest cost-to-IO, or esc_y if no strictly-non-worse downward cell exists (same
# `>= currmin+1` gate as the upward version, so it only fires when the A* map actually
# routes the shortest path around the blocker through a lower row).
function sidestep_down_astar(matrix, blockmat, asternmat, escortid, escorts, items)
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    esc_x, esc_y = escorts[escortid].coords
    blocked = [y for y in esc_y-1:-1:1 if (blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys)]
    reachable_floor = isempty(blocked) ? 1 : maximum(blocked) + 1
    reachable_floor >= esc_y && return esc_y            # immediately blocked below
    currmin = asternmat[esc_x, esc_y]
    best_val, best_y = Inf, esc_y
    for y in esc_y-1:-1:reachable_floor
        if asternmat[esc_x, y] < best_val
            best_val, best_y = asternmat[esc_x, y], y
        end
    end
    return best_val >= currmin + 1 ? esc_y : best_y
end
function freeroam!(iteration, matrix, items, escorts, escortid, blockmat, IO)
    strategy = IO[1] == 1 ? 1 : IO[1] == size(matrix, 1) ? 3 : 2 # 1: left, 2: middle, 3: right

    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords
    avgesc_x = length(keys(escorts)) > 1 ? mean([escorts[esc].coords[1] for esc in keys(escorts) if esc != escortid]) : esc_x
    if strategy ==2 && avgesc_x<IO[1]
        strategy = 3 # if most escorts are on the left we prefer staying as right as possible while moving left
    elseif strategy ==2 && avgesc_x>IO[1]
        strategy = 1# if most escorts are on the right we prefer staying as left as possible while moving right
    end
    moveitnow = false
    if escorts[escortid].lastmoved <= iteration-2 
        moveitnow = true
    end
  
    # FREE ROAM; GO SOMEWHERE ELSE/FREE IF POSSIBLE
    
    worked, asternmat = outwards_astar_with_dirchange(matrix, IO, blockmat,escortid,escorts,items)

    # LM: descend the cost-to-IO map directly instead of the left/up/right ladder.
    # The ladder can only advance along the IO's own row/column toward the IO or retreat
    # perpendicular to it, so it cannot route an idle escort AROUND an item wedged against
    # the IO. asternmat is a true shortest-path cost from the IO around the item obstacles,
    # so every strictly-cheaper neighbour is on a real shortest path — greedy descent is exact.
    # Auto-gated to the lone-escort / multi-load regime (Mirzaei-type), where the ladder
    # deadlocks and the multi-escort pile-up downside cannot occur; FREEROAM_GRADIENT[]
    # forces it on regardless for experimentation.
    lone_escort_multi_load = length(escorts) == 1 && length(items) >= 2
    if UNIT_STEP[] && (FREEROAM_GRADIENT[] || lone_escort_multi_load) &&
       isfinite(asternmat[esc_x, esc_y]) && !(esc_x == IO[1] && esc_y == IO[2])
        _blockers = setdiff(union(keys(escorts), keys(items)), [escortid])
        _bx, _by, _bcost = esc_x, esc_y, asternmat[esc_x, esc_y]
        for (nx, ny) in ((esc_x+1, esc_y), (esc_x-1, esc_y), (esc_x, esc_y+1), (esc_x, esc_y-1))
            if 1 <= nx <= size(matrix, 1) && 1 <= ny <= size(matrix, 2) &&
               blockmat[nx, ny] != 1 && !(matrix[nx, ny] in _blockers) &&
               !((nx, ny) in thisescort.tabu) && asternmat[nx, ny] < _bcost
                _bx, _by, _bcost = nx, ny, asternmat[nx, ny]
            end
        end
        if (_bx, _by) != (esc_x, esc_y)
            return true, (_bx, _by)
        end
    end

    if esc_x == IO[1] && esc_y == IO[2]
        return true, (esc_x, esc_y) # best place it could be 
    elseif esc_x == IO[1] # down, outwards, 
        maxmove_y = checkasternmat(blockmat, matrix, -2, escortid, strategy, escorts, items,IO, asternmat)
        if (maxmove_y == esc_y) || haskey(escorts, matrix[esc_x, maxmove_y ]) || (esc_x, maxmove_y) in thisescort.tabu #cannot move down enough , move out of the way right or left 
            avg_x = mean([items[item].coords[1] for item in keys(items)]) # where are the items ? 
            diresc= avg_x <= IO[1] ? 1 : -1 # if items are left we go right, vice versa
            for _ in 1:2 # Try both directions if the first choice fails
                if diresc == 1 || IO[1] ==1 # chose right
                    maxmove = size(matrix, 1)
                    maxmove = checkmatrixforblock!(blockmat, matrix, diresc, escortid, strategy, iteration, escorts, items, IO)
                    if (maxmove > esc_x && !(haskey(escorts, matrix[maxmove, esc_y])))&& !((maxmove, esc_y) in thisescort.tabu) # can move right
                        return true, (maxmove, esc_y)
                    end
                elseif diresc == -1 || IO[1] == size(matrix,1)# chose left
                    maxmove = 1
                    maxmove = checkmatrixforblock!(blockmat, matrix, diresc, escortid, strategy, iteration, escorts, items,IO)
                    if (maxmove < esc_x && !(haskey(escorts, matrix[maxmove, esc_y])))&& !((maxmove, esc_y) in thisescort.tabu) # can move left
                        return true, (maxmove, esc_y)
                    end
                end
                diresc = -diresc # Switch direction
            end 
            if moveitnow # side is also blocked. so now we couldnt move down or sideways
                minup = checkmatrixforblock!(blockmat, matrix, 2, escortid, strategy, iteration, escorts, items,IO)
                if minup > esc_y
                    return true, (esc_x, minup)
                end
            end

        else
            return true, (esc_x, maxmove_y)
        end
    elseif esc_y == IO[2] # go in X direction towards IO , if blocked go up, if must move go outwards 
        if esc_x < IO[1] # io on the right
            maxmove_x = checkasternmat( blockmat, matrix, 1, escortid, strategy, escorts, items,IO, asternmat)
            if maxmove_x == esc_x || haskey(escorts, matrix[maxmove_x, esc_y]) || (maxmove_x, esc_y) in thisescort.tabu 
                minup = esc_y >= size(matrix, 2) ?
                    sidestep_down_astar(matrix, blockmat, asternmat, escortid, escorts, items) :
                    (asternmat[esc_x,esc_y+1] != Inf ? checkasternmat(blockmat, matrix, 2, escortid, strategy, escorts, items,IO, asternmat) :
                    checkmatrixforblock!(blockmat, matrix, 2, escortid, strategy, iteration, escorts, items,IO))
                if minup != esc_y
                    return true, (esc_x, minup)
                elseif moveitnow 
                    maxmove_x = checkasternmat( blockmat, matrix, -1, escortid, strategy, escorts, items,IO, asternmat)
                    if maxmove_x == esc_x || haskey(escorts, matrix[maxmove_x, esc_y]) || (maxmove_x, esc_y) in thisescort.tabu 
                        return true, (maxmove_x, esc_y)
                    end
                end
            end                
        else # io on the left
            maxmove_x = checkasternmat( blockmat, matrix, -1, escortid, strategy, escorts, items,IO, asternmat)
            if maxmove_x == esc_x || haskey(escorts, matrix[maxmove_x, esc_y]) || (maxmove_x, esc_y) in thisescort.tabu 
                minup = esc_y >= size(matrix, 2) ?
                    sidestep_down_astar(matrix, blockmat, asternmat, escortid, escorts, items) :
                    (asternmat[esc_x,esc_y+1] != Inf ? checkasternmat(blockmat, matrix, 2, escortid, strategy, escorts, items,IO, asternmat) :
                    checkmatrixforblock!(blockmat, matrix, 2, escortid, strategy, iteration, escorts, items,IO))
                if minup != esc_y
                    return true, (esc_x, minup)
                elseif moveitnow 
                    maxmove_x = checkasternmat( blockmat, matrix, 1, escortid, strategy, escorts, items,IO, asternmat)
                    if maxmove_x == esc_x || haskey(escorts, matrix[maxmove_x, esc_y]) || (maxmove_x, esc_y) in thisescort.tabu 
                        return true,(maxmove_x, esc_y)
                    end
                end
            end     
        end  
        return true, (maxmove_x, esc_y)
    else # not at IO coords # down, inwards, upwards, outwards
        avg_x = mean([items[item].coords[1] for item in keys(items)])
        dirx= avg_x <= IO[1] ? 1 : -1 # try go to the opposite direction of the items to be able to serve them
        iodir = dirx
        maxmove_y = checkasternmat(blockmat, matrix, -2, escortid, strategy, escorts, items,IO, asternmat)#checkmatrixforblock!(blockmat, matrix, 1, -2, escortid, strategy, iteration, escorts, items,IO)
        if (maxmove_y == esc_y) || haskey(escorts, matrix[esc_x, maxmove_y ]) || (esc_x, maxmove_y) in thisescort.tabu# cannot move down enough , move out of the way right or left 
            for _ in 1:2 
                if dirx == 1 # chose right
                    maxmove = iodir == dirx ?  # if asternmat can be used we use it
                            checkasternmat( blockmat, matrix, dirx, escortid, strategy, escorts, items,IO, asternmat) :
                            checkmatrixforblock!(blockmat, matrix, dirx, escortid, strategy, iteration, escorts, items,IO)
                    if maxmove > esc_x && !(haskey(escorts, matrix[maxmove, esc_y])) && !((maxmove, esc_y) in thisescort.tabu) # can move right
                        return true, (maxmove, esc_y)
                    end
                elseif dirx == -1 # chose left
                    maxmove = iodir == dirx ? 
                            checkasternmat( blockmat, matrix, dirx, escortid, strategy, escorts, items,IO, asternmat) :
                            checkmatrixforblock!(blockmat, matrix, dirx, escortid, strategy, iteration, escorts, items,IO)
                    if (maxmove < esc_x && !(haskey(escorts, matrix[maxmove, esc_y]))) && !((maxmove, esc_y) in thisescort.tabu) # can move left
                        return true, (maxmove, esc_y)
                    end
                end
                dirx = -dirx
            end
        else # can go down
            return true, (esc_x, maxmove_y)
        end
        if moveitnow # must move so we try up
            minup = checkmatrixforblock!(blockmat, matrix, 2, escortid, strategy, iteration, escorts, items,IO)
            if minup > esc_y
                return true,(esc_x, minup)
            end
        end
        
    end
    
    return false , (esc_x, esc_y ) # cannot move
end
function freeroam_dumb!(iteration, matrix, items, escorts, escortid, blockmat, IO)
    strategy = IO[1] == 1 ? 1 : IO[1] == size(matrix, 1) ? 3 : 2 # 1: left, 2: middle, 3: right

    thisescort = escorts[escortid]
    esc_x, esc_y = thisescort.coords
    avgesc_x = length(keys(escorts)) > 1 ? mean([escorts[esc].coords[1] for esc in keys(escorts) if esc != escortid]) : esc_x
    if strategy ==2 && avgesc_x<IO[1]
        strategy = 3 # if most escorts are on the left we prefer staying as right as possible while moving left
    elseif strategy ==2 && avgesc_x>IO[1]
        strategy = 1# if most escorts are on the right we prefer staying as left as possible while moving right
    end
    moveitnow = false
    if escorts[escortid].lastmoved <= iteration-2 
        moveitnow = true
    end
  
    # FREE ROAM; GO SOMEWHERE ELSE/FREE IF POSSIBLE
    
    worked, asternmat = outwards_astar_with_dirchange(matrix, IO, blockmat,escortid,escorts,items)
    if esc_x == IO[1] && esc_y == IO[2]
        return true, (esc_x, esc_y) # best place it could be 
    elseif esc_x == IO[1] # down, outwards, 
        maxmove_y = checkasternmat(blockmat, matrix, -2, escortid, strategy, escorts, items,IO, asternmat)
        if (maxmove_y == esc_y) || haskey(escorts, matrix[esc_x, maxmove_y ]) || (esc_x, maxmove_y) in thisescort.tabu #cannot move down enough , move out of the way right or left 
            avg_x = mean([items[item].coords[1] for item in keys(items)]) # where are the items ? 
            diresc= avg_x <= IO[1] ? 1 : -1 # if items are left we go right, vice versa
            for _ in 1:2 # Try both directions if the first choice fails
                if diresc == 1 || IO[1] ==1 # chose right
                    maxmove = size(matrix, 1)
                    maxmove = checkmatrixforblock!(blockmat, matrix, diresc, escortid, strategy, iteration, escorts, items, IO)
                    if (maxmove > esc_x && !(haskey(escorts, matrix[maxmove, esc_y])))&& !((maxmove, esc_y) in thisescort.tabu) # can move right
                        return true, (maxmove, esc_y)
                    end
                elseif diresc == -1 || IO[1] == size(matrix,1)# chose left
                    maxmove = 1
                    maxmove = checkmatrixforblock!(blockmat, matrix, diresc, escortid, strategy, iteration, escorts, items,IO)
                    if (maxmove < esc_x && !(haskey(escorts, matrix[maxmove, esc_y])))&& !((maxmove, esc_y) in thisescort.tabu) # can move left
                        return true, (maxmove, esc_y)
                    end
                end
                diresc = -diresc # Switch direction
            end 
            if moveitnow # side is also blocked. so now we couldnt move down or sideways
                minup = checkmatrixforblock!(blockmat, matrix, 2, escortid, strategy, iteration, escorts, items,IO)
                if minup > esc_y
                    return true, (esc_x, minup)
                end
            end

        else
            return true, (esc_x, maxmove_y)
        end
    elseif esc_y == IO[2] # go in X direction towards IO , if blocked go up, if must move go outwards 
        if esc_x < IO[1] # io on the right
            maxmove_x = checkasternmat( blockmat, matrix, 1, escortid, strategy, escorts, items,IO, asternmat)
            if maxmove_x == esc_x || haskey(escorts, matrix[maxmove_x, esc_y]) || (maxmove_x, esc_y) in thisescort.tabu 
                minup = esc_y >= size(matrix, 2) ?
                    sidestep_down_astar(matrix, blockmat, asternmat, escortid, escorts, items) :
                    (asternmat[esc_x,esc_y+1] != Inf ? checkasternmat(blockmat, matrix, 2, escortid, strategy, escorts, items,IO, asternmat) :
                    checkmatrixforblock!(blockmat, matrix, 2, escortid, strategy, iteration, escorts, items,IO))
                if minup != esc_y
                    return true, (esc_x, minup)
                elseif moveitnow 
                    maxmove_x = checkasternmat( blockmat, matrix, -1, escortid, strategy, escorts, items,IO, asternmat)
                    if maxmove_x == esc_x || haskey(escorts, matrix[maxmove_x, esc_y]) || (maxmove_x, esc_y) in thisescort.tabu 
                        return true, (maxmove_x, esc_y)
                    end
                end
            end                
        else # io on the left
            maxmove_x = checkasternmat( blockmat, matrix, -1, escortid, strategy, escorts, items,IO, asternmat)
            if maxmove_x == esc_x || haskey(escorts, matrix[maxmove_x, esc_y]) || (maxmove_x, esc_y) in thisescort.tabu 
                minup = esc_y >= size(matrix, 2) ?
                    sidestep_down_astar(matrix, blockmat, asternmat, escortid, escorts, items) :
                    (asternmat[esc_x,esc_y+1] != Inf ? checkasternmat(blockmat, matrix, 2, escortid, strategy, escorts, items,IO, asternmat) :
                    checkmatrixforblock!(blockmat, matrix, 2, escortid, strategy, iteration, escorts, items,IO))
                if minup != esc_y
                    return true, (esc_x, minup)
                elseif moveitnow 
                    maxmove_x = checkasternmat( blockmat, matrix, 1, escortid, strategy, escorts, items,IO, asternmat)
                    if maxmove_x == esc_x || haskey(escorts, matrix[maxmove_x, esc_y]) || (maxmove_x, esc_y) in thisescort.tabu 
                        return true,(maxmove_x, esc_y)
                    end
                end
            end     
        end  
        return true, (maxmove_x, esc_y)
    else # not at IO coords # down, inwards, upwards, outwards
        avg_x = mean([items[item].coords[1] for item in keys(items)])
        dirx= avg_x <= IO[1] ? 1 : -1 # try go to the opposite direction of the items to be able to serve them
        iodir = dirx
        maxmove_y = checkmatrixforblock!(blockmat, matrix, -2, escortid, strategy, iteration, escorts, items,IO)#checkmatrixforblock!(blockmat, matrix, 1, -2, escortid, strategy, iteration, escorts, items,IO)
        if (maxmove_y == esc_y) || haskey(escorts, matrix[esc_x, maxmove_y ]) || (esc_x, maxmove_y) in thisescort.tabu# cannot move down enough , move out of the way right or left 
            for _ in 1:2 
                if dirx == 1 # chose right
                    maxmove = iodir == dirx ?  # if asternmat can be used we use it
                            checkasternmat( blockmat, matrix, dirx, escortid, strategy, escorts, items,IO, asternmat) :
                            checkmatrixforblock!(blockmat, matrix, dirx, escortid, strategy, iteration, escorts, items,IO)
                    if maxmove > esc_x && !(haskey(escorts, matrix[maxmove, esc_y])) && !((maxmove, esc_y) in thisescort.tabu) # can move right
                        return true, (maxmove, esc_y)
                    end
                elseif dirx == -1 # chose left
                    maxmove = iodir == dirx ? 
                            checkasternmat( blockmat, matrix, dirx, escortid, strategy, escorts, items,IO, asternmat) :
                            checkmatrixforblock!(blockmat, matrix, dirx, escortid, strategy, iteration, escorts, items,IO)
                    if (maxmove < esc_x && !(haskey(escorts, matrix[maxmove, esc_y]))) && !((maxmove, esc_y) in thisescort.tabu) # can move left
                        return true, (maxmove, esc_y)
                    end
                end
                dirx = -dirx
            end
        else # can go down
            return true, (esc_x, maxmove_y)
        end
        if moveitnow # must move so we try up
            minup = checkmatrixforblock!(blockmat, matrix, 2, escortid, strategy, iteration, escorts, items,IO)
            if minup > esc_y
                return true,(esc_x, minup)
            end
        end
        
    end
    
    return false , (esc_x, esc_y ) # cannot move
end
function checkasternmat(blockmat, matrix, diresc, escortid, strategy, escorts, items, IO,asternmat)
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    esc_x, esc_y = escorts[escortid].coords
    
    if diresc == 1 
        valid_x_1 = [x for x in esc_x:size(matrix,1) if (blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys)] # right
        if (isempty(valid_x_1) || minimum(valid_x_1) > esc_x+1)
            maxmove = isempty(valid_x_1) ?  size(matrix,1) : minimum(valid_x_1)-1
            valy = asternmat[esc_x, esc_y]+1 ;currmin = asternmat[esc_x, esc_y]
            if strategy == 1 # going right, we prefer as left as possible 
                for x in esc_x:maxmove
                    currval = asternmat[x, esc_y]
                    if currval < valy
                        maxmove = x
                        valy = currval
                    end
                end
            else
                for x in maxmove:-1:esc_x # backwards, as we want to move as far right as we can 
                    currval = asternmat[x, esc_y]
                    if currval < valy
                        maxmove = x
                        valy = currval
                    end
                end
            end 
            if valy >= currmin +1
                return esc_x
            else
                return maxmove
            end
        else 
            return esc_x
        end
    end
    if diresc == -1 # going left checking 
        valid_x_1m = [x for x in esc_x:-1:1 if (blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys)] # left
        if (isempty(valid_x_1m) || maximum(valid_x_1m) < esc_x-1) # can move left
            maxmove = isempty(valid_x_1m) ? 1 : (maximum(valid_x_1m)+1)
            valy = asternmat[esc_x, esc_y]+1 ;currmin = asternmat[esc_x, esc_y]
            if strategy == 3 # prefer near rightside IO
                for x in esc_x:-1:maxmove
                    currval = asternmat[x, esc_y]
                    if currval < valy
                        maxmove = x
                        valy = currval
                    end
                end
            else
                for x in maxmove:esc_x # prefer far out
                    currval = asternmat[x, esc_y]
                    if currval < valy
                        maxmove = x
                        valy = currval
                    end
                end
            end
            if valy >= currmin +1
                return esc_x
            else
                return maxmove
            end
        else 
            return esc_x
        end
    end
    if diresc == 2# lowest possible y
        valid_y= [y for y in esc_y:size(matrix,2) if (blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys)] # up 
        if isempty(valid_y) || minimum(valid_y) - 1 > esc_y # can move up
            minup = isempty(valid_y) ? size(matrix, 2) : minimum(valid_y) - 1 
            min_val = Inf
            min_idx = esc_y
            currmin = asternmat[esc_x, esc_y]
            for y in esc_y:minup 
                if asternmat[esc_x, y] < min_val && y != esc_y
                    min_val = asternmat[esc_x, y]
                    min_idx = y
                end
            end
            if min_val >= currmin +1
                return esc_y
            else
                return min_idx
            end
        else 
            return esc_y
        end
    end
    if diresc == -2 # lowest possible y
        valid_y= [y for y in IO[2]:esc_y-1 if (blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys)] # up 
        if (isempty(valid_y) || maximum(valid_y)+1 < esc_y) # can move up
            range1 = isempty(valid_y) ? IO[2] : maximum(valid_y)+1
            min_val, min_idx = Inf, esc_y; currmin = asternmat[esc_x, esc_y]
            for y in range1:esc_y
                if asternmat[esc_x, y] < min_val
                    min_val = asternmat[esc_x, y]
                    min_idx = y
                end
            end
            if min_val >= currmin +1
                return esc_y
            else
                return min_idx
            end
        else 
            return esc_y
        end
    end
    
    
    
    return maxmove

end
function checkmatrixforblock!(blockmat, matrix, diresc, escortid, strategy, iteration, escorts, items, IO)
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    esc_x, esc_y = escorts[escortid].coords
    
    if diresc == 1 
        valid_x_1 = [x for x in esc_x:size(matrix,1) if (blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys)] # right
        if (isempty(valid_x_1) || minimum(valid_x_1) > esc_x+1)
            maxmove = isempty(valid_x_1) ?  size(matrix,1) : minimum(valid_x_1)-1
            valy = esc_y
            if strategy == 1 # prefer near leftside IO
                for x in esc_x:maxmove
                    candheight = directionval(matrix, items, escorts, escortid, x, esc_y, 2, iteration,IO) 
                    if candheight < valy
                        maxmove = x
                        valy = candheight
                    end
                end
            else
                for x in maxmove:-1:esc_x # backwards, as we want to move as far right as we can 
                    candheight = directionval(matrix, items, escorts, escortid, x, esc_y, 2, iteration,IO) 
                    if candheight < valy
                        maxmove = x
                        valy = candheight
                    end
                end
            end
            return maxmove
        else 
            return esc_x
        end
    end
    if diresc == -1 # going left checking 
        valid_x_1m = [x for x in esc_x:-1:1 if (blockmat[x, esc_y] == 1 || matrix[x, esc_y] in allkeys)] # left
        if (isempty(valid_x_1m) || maximum(valid_x_1m) < esc_x-1) # can move left
            maxmove = isempty(valid_x_1m) ? 1 : (maximum(valid_x_1m)+1)
            valy = esc_y
            if strategy == 3 # prefer near rightside IO
                for x in esc_x:-1:maxmove
                    candheight = directionval(matrix, items, escorts, escortid, x, esc_y, 2, iteration,IO) 
                    if candheight < valy
                        maxmove = x
                        valy = candheight
                    end
                end
            else
                for x in maxmove:esc_x # prefer far out
                    candheight = directionval(matrix, items, escorts, escortid, x, esc_y, 2, iteration,IO) 
                    if candheight < valy
                        maxmove = x
                        valy = candheight
                    end
                end
            end
            return maxmove
        else 
            return esc_x
        end
    end
    if diresc == 2# lowest possible y
        valid_y= [y for y in esc_y:size(matrix,2) if (blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys)] # up 
        if (isempty(valid_y) || minimum(valid_y)-1 > esc_y) # can move up
            minup = isempty(valid_y) ? size(matrix, 2) : minimum(valid_y)-1 
            for y in esc_y:minup
                if directionclear(items, escorts,escortid, esc_x,y, 1, iteration,IO)
                    minup = y
                    break
                end
            end
            return minup
        else 
            return esc_y
        end
    end
    if diresc == -2 # lowest possible y
        reach = 2 # reach is the left right checking reach in this case. 
        valid_y= [y for y in IO[2]:esc_y-1 if (blockmat[esc_x, y] == 1 || matrix[esc_x, y] in allkeys)] # up 
        if (isempty(valid_y) || maximum(valid_y)+1 < esc_y) # can move up
            range1 = isempty(valid_y) ? IO[2] : maximum(valid_y)+1
            maxdown = esc_y
            for y in range1:esc_y
                if directionclear(items, escorts,escortid, esc_x,y, 1, reach, iteration,IO)
                    maxdown = y
                    break
                end
            end
            return maxdown
        else 
            return esc_y
        end
    end
    
    
    
    return maxmove

end

function directionval( matrix, items, escorts, escortid, coordx, coordy, direction, iteration,IO)
    iox, ioy = IO
    valy = 1 ; valx = 1
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    if direction ==1 # we give y coord and tell to check 
        for entity in allkeys # check for presence in both collections in the front          
            if haskey(escorts, entity)
                esc_other = escorts[entity]
                if (esc_other.coords[2] == coordy && esc_other.lastmoved == iteration)  # iteration as escort will probably move
                    if (esc_other.coords[1] < min(iox, coordx) || esc_other.coords[1] > max(iox, coordx)) # acceptable if on the other side of iox
                        continue
                    end
                    return false
                end
            elseif haskey(items, entity)
                item_other = items[entity]
                if item_other.coords[2] == coordy
                    if (coordx == iox) ||
                        (coordx > min(item_other.coords[1], iox) && coordx < max(item_other.coords[1], iox))
                        #println(" item:$(entity) can be served by escort:$escortid , shouldnt have happened unless path blocked after move")
                        continue
                    end
                    return false
                end
            end
        end
        return valx
    elseif direction ==2 # we give x coord and tell to check lower y, as we wish to not hurt anything
        if coordy>1
            for y in coordy-1:-1:1
                if matrix[coordx, y] in allkeys
                    valy = y+1
                end
            end
        end
        return valy
    end
    
end
"""
while we check where to move the escort so that we can make another move in the next iteration 

    example: I want to move escort up as its blocked here. i want to check where up can i move so that i can move left/right in the next iteration
"""
function directionclear( items, escorts, escortid, coordx, coordy, direction, iteration,IO)
    iox, ioy = IO
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    if direction ==1 # we give y coord and tell to check 
        for entity in allkeys # check for presence in both collections in the front          
            if haskey(escorts, entity)
                esc_other = escorts[entity]
                if (esc_other.coords[2] == coordy && esc_other.lastmoved == iteration)  # iteration as escort will probably move
                    if (esc_other.coords[1] < min(iox, coordx) || esc_other.coords[1] > max(iox, coordx)) # acceptable if on the other side of iox
                        continue
                    end
                    return false
                end
            elseif haskey(items, entity)
                item_other = items[entity]
                if item_other.coords[2] == coordy
                    if (coordx == iox) ||
                        (coordx > min(item_other.coords[1], iox) && coordx < max(item_other.coords[1], iox))
                        #println(" item:$(entity) can be served by escort:$escortid , shouldnt have happened unless path blocked after move")
                        continue
                    end
                    return false
                end
            end
        end
        
    elseif direction ==2 # we give x coord and tell to check lower y, as we wish to not hurt anything
        for entity in allkeys # check for presence in both collections in the front
            if haskey(escorts, entity)
                esc_other = escorts[entity]
                if (esc_other.coords[1] == coordx && esc_other.lastmoved == iteration) # iteration as escort will probably move
                    return false
                end
            elseif haskey(items, entity)
                item_other = items[entity]
                if item_other.coords[1] == coordx && item_other.coords[2] < coordy 
                    return false
                end
            end
        end
    end
    return true
end

function directionclear( items, escorts, escortid, coordx, coordy, direction, reach, iteration,IO)
    iox, ioy = IO
    dir = iox < coordx ? -1 : 1
    dir = iox == coordx ? 0 : dir
    allkeys = setdiff(union(keys(escorts), keys(items)), [escortid])
    if direction ==1 # we give y coord and tell to check 
        for entity in allkeys # check for presence in both collections in the front          
            if haskey(escorts, entity)
                esc_other = escorts[entity]
                if (esc_other.coords[2] == coordy && esc_other.lastmoved == iteration)  # iteration as escort will probably move
                    if (esc_other.coords[1] < min(iox, coordx) || esc_other.coords[1] > max(iox, coordx)) # acceptable if on the other side of iox
                        continue
                    elseif (esc_other.coords[1] >= (coordx - reach) && esc_other.coords[1] <= (coordx + reach))
                        return false
                    end
                end
            elseif haskey(items, entity)
                item_other = items[entity]
                if (
                    item_other.coords[2] == coordy &&
                    dir == -1 &&
                    item_other.coords[1] >= (coordx - reach) &&
                    item_other.coords[1] <= coordx
                ) ||
                (
                    item_other.coords[2] == coordy &&
                    dir == 1 &&
                    item_other.coords[1] >= coordx &&
                    item_other.coords[1] <= (coordx + reach)
                )
                    
                    return false
                end
            end
        end
        
    elseif direction ==2 # we give x coord and tell to check lower y, as we wish to not hurt anything
        for entity in allkeys # check for presence in both collections in the front
            if haskey(escorts, entity)
                esc_other = escorts[entity]
                if (esc_other.coords[1] == coordx  &&
                     esc_other.coords[2] >=(coordy - reach) && esc_other.coords[2] <= (coordy))  # iteration as escort will probably move
                    return false
                end
            elseif haskey(items, entity)
                item_other = items[entity]
                if item_other.coords[1] == coordx && 
                    item_other.coords[2] >= coordy-reach &&
                    item_other.coords[2] <= coordy + reach   
                    return false
                end
            end
        end
    end
    return true

end
"""
some items on the way may also be moved (doubleserve) and the doubleserve migh block IO
"""
function generatefuturecoords(items,  escorts,dir, escortid, itemid, matrix, IO) 
    item = items[itemid]
    itemx, itemy = item.coords
    esc_x, esc_y = escorts[escortid].coords
    itemscoords = Tuple{Int,Int}[]
    if dir==2
        for o_itemid in keys(items)
            if o_itemid == itemid
                continue
            else
                o_item = items[o_itemid]
                o_itemx, o_itemy = o_item.coords
                if o_itemx == itemx && o_itemy < itemy && o_itemy > esc_y
                    push!(itemscoords, (o_itemx, (max(1,(o_itemy-1)))))
                else
                    push!(itemscoords, (o_itemx, o_itemy))
                end
            end
        end
        push!(itemscoords,(itemx, (max(1,(itemy-1)))))
    elseif dir==1
        futurecoords = (itemx, itemy)
        dir = IO[1] < itemx ? -1 : 1
        if dir == 1
            futurecoords = (min(itemx+1,size(matrix, 1)), itemy)
        else
            futurecoords = (max(itemx-1,1), itemy)
        end
        for o_itemid in keys(items)
            if o_itemid == itemid
                continue
            else
                o_item = items[o_itemid]
                o_itemx, o_itemy = o_item.coords
                if o_itemy == itemy &&(( o_itemx < itemx && o_itemx > esc_x) || (o_itemx > itemx && esc_x > o_itemx))
                    push!(itemscoords, (o_itemx+dir, o_itemy))
                else
                    push!(itemscoords, (o_itemx, o_itemy))
                end
            end
        end
        push!(itemscoords, futurecoords)
    end
    return itemscoords
end

"""
Multi-IO version of generatefuturecoords: handles items targeting different IOs
Each item uses its own target IO to determine movement direction
"""
function generatefuturecoords_multi_io(items, escorts, dir, escortid, itemid, matrix, item_to_ios, current_io)
    item = items[itemid]
    itemx, itemy = item.coords
    esc_x, esc_y = escorts[escortid].coords
    itemscoords = Tuple{Int,Int}[]
    
    if dir == 2  # y-direction movement
        for o_itemid in keys(items)
            if o_itemid == itemid
                continue
            else
                o_item = items[o_itemid]
                o_itemx, o_itemy = o_item.coords
                
                # Check if item is in vertical path
                if o_itemx == itemx && o_itemy < itemy && o_itemy > esc_y
                    push!(itemscoords, (o_itemx, (max(1, (o_itemy - 1)))))
                else
                    push!(itemscoords, (o_itemx, o_itemy))
                end
            end
        end
        # Main item moves toward its target IO
        push!(itemscoords, (itemx, (max(1, (itemy - 1)))))
        
    elseif dir == 1  # x-direction movement
        # Use current_io to determine direction for main item
        dir_sign = current_io[1] < itemx ? -1 : 1
        futurecoords = (itemx, itemy)
        
        if dir_sign == 1
            futurecoords = (min(itemx + 1, size(matrix, 1)), itemy)
        else
            futurecoords = (max(itemx - 1, 1), itemy)
        end
        
        # Process other items - use their target IOs
        for o_itemid in keys(items)
            if o_itemid == itemid
                continue
            else
                o_item = items[o_itemid]
                o_itemx, o_itemy = o_item.coords
                
                # Check if item is in horizontal path
                if o_itemy == itemy && ((o_itemx < itemx && o_itemx > esc_x) || (o_itemx > itemx && esc_x > o_itemx))
                    # Determine direction for this other item based on its target IO
                    if haskey(item_to_ios, o_itemid) && !isempty(item_to_ios[o_itemid])
                        # Use first target IO for other items
                        other_target_io = item_to_ios[o_itemid][1]
                        other_dir_sign = other_target_io[1] < o_itemx ? -1 : 1
                        if !(dir_sign == other_dir_sign)
                            println("Warning: Item $o_itemid has a different target IO direction than the main item. Defaulting to main item's direction.")
                            other_dir_sign = dir_sign  # Override to ensure consistent movement direction
                        end
                        push!(itemscoords, (o_itemx + other_dir_sign, o_itemy))
                    else
                        # Fallback to same direction as main item if no IO info
                        push!(itemscoords, (o_itemx + dir_sign, o_itemy))
                    end
                else
                    push!(itemscoords, (o_itemx, o_itemy))
                end
            end
        end
        push!(itemscoords, futurecoords)
    end
    
    return itemscoords
end
function generatefuturecoords_fincoord(items,  escorts, dir, escortid, finalcoords, matrix, IO) 
    finx, finy = finalcoords
    esc_x, esc_y = escorts[escortid].coords
    itemscoords = Tuple{Int,Int}[]
    if dir==2
        for o_itemid in keys(items)
          
            o_item = items[o_itemid]
            o_itemx, o_itemy = o_item.coords
            if o_itemx == finx && o_itemy <= finy && finy >= esc_y
                push!(itemscoords, (o_itemx, (max(1,(o_itemy-1)))))
            else
                push!(itemscoords, (o_itemx, o_itemy))
            end
        
        end
    elseif dir==1
        dir = finx < esc_x ? 1 : -1 # careful about direction
        for o_itemid in keys(items)
        
            o_item = items[o_itemid]
            o_itemx, o_itemy = o_item.coords
            if o_itemy == finy && ((esc_x <= o_itemx && o_itemx <= finx ) || (esc_x >= o_itemx && o_itemx >= finx))
                push!(itemscoords, (o_itemx+dir, o_itemy))
            else
                push!(itemscoords, (o_itemx, o_itemy))
            end
           
        end
    end
    return itemscoords
end
function futurecoords_closetoIO(items,  itemid, escorts, escortid, dir, IO) 
    finx, finy = items[itemid].coords
    esc_x, esc_y = escorts[escortid].coords
    iox,ioy = IO
    itemscounter = 0
    if dir==2
        for o_itemid in keys(items)
          
            o_item = items[o_itemid]
            o_itemx, o_itemy = o_item.coords
            if o_itemx == finx && o_itemy <= finy && finy >= esc_y
                currcoords = (o_itemx, (max(1,(o_itemy-1))))
                if abs(currcoords[1] -iox) + abs(currcoords[2] -ioy) <= length(keys(items))-1
                    itemscounter += 1
                end
            elseif abs(o_itemx -iox) + + abs(o_itemy -ioy) <= length(keys(items))-1
                itemscounter += 1
            end
        end
    elseif dir==1
        dir = finx < esc_x ? 1 : -1 # careful about direction
        for o_itemid in keys(items)
    
            o_item = items[o_itemid]
            o_itemx, o_itemy = o_item.coords
            if o_itemy == finy && ((esc_x <= o_itemx && o_itemx <= finx ) || (esc_x >= o_itemx && o_itemx >= finx))
                currcoords =  (o_itemx+dir, o_itemy)
                if abs(currcoords[1] -iox) + abs(currcoords[2] -ioy) <= length(keys(items))-1
                    itemscounter += 1
                end
            elseif abs(o_itemx -iox) + + abs(o_itemy -ioy) <= length(keys(items))-1
                itemscounter += 1
            end
           
        end
    end
    if itemscounter >= length(keys(items))-1
        return true
    else
        return false
    end
end
"""
returns individual matrices for each urgent customer to figure out how to reach them
"""
function urgmats(items, escorts, blockmat, matrix, urgentcustomers, IO)
    allkeys= union(keys(escorts), keys(items))
    urgmats = Dict{String, Matrix{Int}}()
    if !isempty(urgentcustomers)# Then, sort them by urgency:
        for customer_id in urgentcustomers
            urgmat = deepcopy(blockmat)
            urgx, urgy = items[customer_id].coords
            urgmat[urgx, urgy] = 3
            dir = urgx > IO[1] ? -1 : 1
            if urgx != IO[1]
                if dir == 1
                    for xx in min(urgx+1, size(matrix, 1)):IO[1]
                        if !(matrix[xx, urgy] in allkeys)
                            for y in min(urgy+1, size(matrix,2)):size(matrix, 2)
                                if blockmat[xx, y] == 0 && !(haskey(items, matrix[xx, y]))
                                    urgmat[xx, y] = 2
                                else
                                    break
                                end
                            end
                        else
                            break
                        end
                    end
                else
                    for xx in max(urgx-1, 1):-1:IO[1]
                        if !(matrix[xx, urgy] in allkeys)
                            for y in min(urgy+1, size(matrix,2)):size(matrix, 2)
                                if blockmat[xx, y] == 0 && !(haskey(items, matrix[xx, y]))
                                    urgmat[xx, y] = 2
                                else
                                    break
                                end
                            end
                        else
                            break
                        end
                    end
                end
            end
            if urgy != IO[2]
                for yy in urgy-1:-1:IO[2]
                    if !(matrix[urgx, yy] in allkeys)

                        if dir == -1 || urgx == IO[1]# check rightside
                            for x in min(urgx+1, size(matrix, 1)):size(matrix, 1)
                                if blockmat[x, yy] == 0 && !(haskey(items, matrix[x, yy]))
                                    urgmat[x, yy] = 2
                                else
                                    break
                                end
                            end
                        end
                        if dir == 1 || urgx == IO[1]# check leftside
                            for x in max(urgx-1, 1):-1:IO[1]
                                if blockmat[x, yy] == 0 && !(haskey(items, matrix[x, yy]))
                                    urgmat[x, yy] = 2
                                else
                                    break
                                end
                            end
                        end
                        
                    else
                        break
                    end
                end
            end

            #print_matrix(urgmat) ; print_matrix(matrix, blockmat)
            urgmats[customer_id] = urgmat
        end
    end

    return urgmats
end
function urgmats_multi_io(items, escorts, blockmat, matrix, urgentcustomers, item_to_ios, all_ios)
    allkeys = union(keys(escorts), keys(items))
    result = Dict{String, Matrix{Int}}()
    for customer_id in urgentcustomers
        urgmat = deepcopy(blockmat)
        urgx, urgy = items[customer_id].coords
        urgmat[urgx, urgy] = 3

        # Each item uses its own assigned IO instead of a global one
        assigned = get(item_to_ios, customer_id, all_ios)
        item_io = argmin(io -> abs(io[1] - urgx) + abs(io[2] - urgy), assigned)
        iox, ioy = item_io

        dir = urgx > iox ? -1 : 1
        if urgx != iox
            if dir == 1
                for xx in min(urgx+1, size(matrix, 1)):iox
                    if !(matrix[xx, urgy] in allkeys)
                        for y in min(urgy+1, size(matrix,2)):size(matrix, 2)
                            if blockmat[xx, y] == 0 && !(haskey(items, matrix[xx, y]))
                                urgmat[xx, y] = 2
                            else
                                break
                            end
                        end
                    else
                        break
                    end
                end
            else
                for xx in max(urgx-1, 1):-1:iox
                    if !(matrix[xx, urgy] in allkeys)
                        for y in min(urgy+1, size(matrix,2)):size(matrix, 2)
                            if blockmat[xx, y] == 0 && !(haskey(items, matrix[xx, y]))
                                urgmat[xx, y] = 2
                            else
                                break
                            end
                        end
                    else
                        break
                    end
                end
            end
        end
        if urgy != ioy
            for yy in urgy-1:-1:ioy
                if !(matrix[urgx, yy] in allkeys)
                    if dir == -1 || urgx == iox
                        for x in min(urgx+1, size(matrix, 1)):size(matrix, 1)
                            if blockmat[x, yy] == 0 && !(haskey(items, matrix[x, yy]))
                                urgmat[x, yy] = 2
                            else
                                break
                            end
                        end
                    end
                    if dir == 1 || urgx == iox
                        for x in max(urgx-1, 1):-1:iox
                            if blockmat[x, yy] == 0 && !(haskey(items, matrix[x, yy]))
                                urgmat[x, yy] = 2
                            else
                                break
                            end
                        end
                    end
                else
                    break
                end
            end
        end
        result[customer_id] = urgmat
    end
    return result
end

function resetitems!(items)
    for id in keys(items)
        items[id].escortsx = Vector{String}()
        items[id].escortsy = Vector{String}()
        items[id].direction = 0 
    end
end
function resetescorts!(escorts, iteration)
    for escort_id in keys(escorts) # reset the serving of items. 
        escort = escorts[escort_id]
        escort.itemsx = Vector{String}()
        escort.itemsy = Vector{String}()
        # Remove keys from banset that are smaller than the current iteration
        for key in keys(escort.banset)
            if key < iteration-4
            delete!(escort.banset, key)
            end
        end
        while length(escort.tabu)>2
            escort.tabu = escort.tabu[2:end]
        end
        if escort.lastmoved <= iteration-3
            escort.tabu = Vector{Tuple{Int64, Int64}}()
        end
    end
end
function allowedOrder(esc_x, io_x, xx, itemx) # double serve check 
    return (
        (esc_x <= io_x < xx < itemx) ||
        (esc_x >= io_x > xx > itemx) ||
        (io_x <= esc_x < xx < itemx) ||
        (io_x >= esc_x > xx > itemx)
    )
end
function allowedOrder(esc_x, iox, itemx) # escort can serve
    return (
        (esc_x < itemx && iox < itemx) ||
        (esc_x > itemx && iox > itemx)
    )
end
# A row move of an escort to itemx shifts every cell in between one step back
# towards the escort's start. If the escort is on the other side of the IO,
# any other target load between the IO and the escort would be pushed away
# from the IO (and the next assignment would push it back: an endless loop).
function hurts_load_across_io(items, itemkey, esc_x, esc_y, io_x, itemx)
    lo, hi = if itemx < io_x < esc_x
        io_x, esc_x
    elseif esc_x < io_x < itemx
        esc_x, io_x
    else
        return false
    end
    return any(k != itemkey && it.coords[2] == esc_y && lo < it.coords[1] < hi for (k, it) in items)
end

function print_matrix(matrix)
    printmat = true
    if printmat 
        nrows, ncols = size(matrix)
        println("Matrix: ")
        # Print the matrix in a transposed manner
        for row in nrows:-1:1
            for col in 1:ncols
                element = string(matrix[col, row])
                print(lpad(element, 6), " ")
            end
            println()
            #println()
        end
    end
end
function print_matrix(matrix, blockmat)
    printmat = false
    if printmat 
        nrows, ncols = size(matrix)
        println("Matrix: ")
        # Print the matrix in a transposed manner
        for row in nrows:-1:1
            for col in 1:ncols
                element = string(matrix[col, row])
                print(lpad(element, 6), " ")
            end
            print("   |   ")
            for col in 1:ncols
                element = string(blockmat[col, row])
                print(lpad(element, 6), " ")
            end
            println()
            #println()
        end
        println("End of print")
    end
end
function checksync(matrix, escorts, items, step)
    for eid in keys(escorts)
        ex, ey = escorts[eid].coords
        if matrix[ex, ey] != eid
            println("$step Escort $eid is not in the right place")
        end
    end
    for iid in keys(items)
        ix, iy = items[iid].coords
        if matrix[ix, iy] != iid
            println("$step Item $iid is not in the right place")
        end
    end
end

# ─── A* from escort → IO (used by cooperative_freeroam!) ─────────────────────
function _escort_astar(
        start         :: Tuple{Int,Int},
        IO            :: Tuple{Int,Int},
        blockmat      :: AbstractMatrix,
        extra_blocked :: Set{Tuple{Int,Int}},
        rows          :: Int,
        cols          :: Int
    ) :: Tuple{Vector{Tuple{Int,Int}}, Float64}

    sx, sy = start
    gx, gy = IO

    dist     = fill(Inf, rows, cols, 3)
    has_from = fill(false, rows, cols, 3)
    from     = Array{Tuple{Int,Int,Int}}(undef, rows, cols, 3)

    heap    = BinaryMinHeap{Tuple{Float64,Int,Int,Int}}()
    visited = fill(false, rows, cols, 3)

    h(x, y) = Float64(abs(x - gx) + abs(y - gy))

    dist[sx, sy, 1] = 0.0
    push!(heap, (h(sx, sy), sx, sy, 1))

    while !isempty(heap)
        (_, cx, cy, cd) = pop!(heap)
        visited[cx, cy, cd] && continue
        visited[cx, cy, cd] = true

        if cx == gx && cy == gy
            path = Tuple{Int,Int}[]
            x, y, d = cx, cy, cd
            while true
                push!(path, (x, y))
                !has_from[x, y, d] && break
                x, y, d = from[x, y, d]
            end
            return reverse!(path), dist[cx, cy, cd]
        end

        for (nx, ny) in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1))
            (1 ≤ nx ≤ rows && 1 ≤ ny ≤ cols) || continue
            blockmat[nx, ny] == 1             && continue
            (nx, ny) ∈ extra_blocked          && continue

            nd   = (nx == cx) ? 3 : 2
            turn = (cd == 1 || cd == nd) ? 0.0 : 1.0
            g    = dist[cx, cy, cd] + 1.0 + turn

            if g < dist[nx, ny, nd]
                dist[nx, ny, nd]     = g
                from[nx, ny, nd]     = (cx, cy, cd)
                has_from[nx, ny, nd] = true
                push!(heap, (g + h(nx, ny), nx, ny, nd))
            end
        end
    end

    return Tuple{Int,Int}[], Inf
end

# ─── Cooperative freeroam: plans all nonmover steps jointly ──────────────────
# Drop-in replacement for the per-escort freeroam! loop in moveescorts_flow!.
# Escorts closer to IO plan first and claim cells; farther escorts route around.
# Each round finds the globally best multi-step move across all remaining escorts,
# executes it, recomputes asternmats, and repeats until no escort can improve.
# checkasternmat allows multi-cell moves in one direction (same as freeroam!).
# Shared reserved set prevents two escorts targeting the same cell.
function cooperative_freeroam!(iteration, matrix, items, escorts, nonmovers, blockmat, IO)
    isempty(nonmovers) && return 0

    rows, cols = size(matrix)
    iox, ioy = IO
    mdist(x, y) = abs(x - iox) + abs(y - ioy)

    reserved = Set{Tuple{Int,Int}}(items[iid].coords for iid in keys(items))
    for eid in keys(escorts)
        eid ∈ nonmovers && continue
        push!(reserved, escorts[eid].coords)
    end

    order = shuffle(Random.default_rng(), collect(nonmovers))
    moved_count = 0

    # compute asternmat once; only recompute when blockmat changes (after a move)
    proxy = first(order)
    worked, asternmat = outwards_astar_with_dirchange(matrix, IO, blockmat, proxy, escorts, items)
    dirty = false

    for eid in order
        esc = escorts[eid]
        sx, sy = esc.coords

        if dirty
            worked, asternmat = outwards_astar_with_dirchange(matrix, IO, blockmat, eid, escorts, items)
            dirty = false
        end

        strategy = IO[1] == 1 ? 1 : IO[1] == size(matrix, 1) ? 3 : 2
        avgesc_x = length(keys(escorts)) > 1 ? mean([escorts[e].coords[1] for e in keys(escorts) if e != eid]) : sx
        if strategy == 2 && avgesc_x < iox; strategy = 3; end
        if strategy == 2 && avgesc_x > iox; strategy = 1; end

        cur_val = worked ? asternmat[sx, sy] : mdist(sx, sy)
        moveitnow = escorts[eid].lastmoved <= iteration - 2

        avg_x = isempty(items) ? sx : mean([items[iid].coords[1] for iid in keys(items)])
        lateral  = avg_x <= iox ? 1 : -1

        dirs = if sx == iox
            (-2, lateral, -lateral, 2)
        elseif sy == ioy
            (sx < iox ? 1 : -1, 2, sx < iox ? -1 : 1, -2)
        else
            (-2, lateral, -lateral, 2)
        end

        best = (sx, sy)

        for dir in dirs
            result = checkasternmat(blockmat, matrix, dir, eid, strategy, escorts, items, IO, asternmat)
            tx = (dir == 1 || dir == -1) ? result : sx
            ty = (dir == 2 || dir == -2) ? result : sy

            (tx, ty) == (sx, sy)           && continue
            (tx, ty) ∈ reserved            && continue
            (tx, ty) ∈ esc.tabu            && continue
            matrix[tx, ty] ∈ keys(escorts) && continue

            val = worked ? asternmat[tx, ty] : mdist(tx, ty)
            val >= cur_val                  && continue

            best = (tx, ty)
            break
        end

        if best == (sx, sy) && moveitnow
            for dir in (-2, lateral, -lateral, 2)
                result = checkmatrixforblock!(blockmat, matrix, dir, eid, strategy, iteration, escorts, items, IO)
                tx = (dir == 1 || dir == -1) ? result : sx
                ty = (dir == 2 || dir == -2) ? result : sy

                (tx, ty) == (sx, sy)           && continue
                (tx, ty) ∈ reserved            && continue
                (tx, ty) ∈ esc.tabu            && continue
                matrix[tx, ty] ∈ keys(escorts) && continue

                best = (tx, ty)
                break
            end
        end

        best == (sx, sy) && (push!(reserved, (sx, sy)); continue)

        tx, ty = best
        nudged = nudge_from_escorts(sx, sy, tx, ty, escorts, eid, blockmat, matrix)
        if nudged ∉ reserved && blockmat[nudged[1], nudged[2]] == 0
            tx, ty = nudged
        end

        push!(esc.tabu, (sx, sy))
        move_escort!(matrix, items, escorts, eid, (tx, ty))
        updateblockmat_e!(blockmat, sx, sy, tx, ty)
        escorts[eid].lastmoved = iteration
        moved_count += 1
        push!(reserved, (tx, ty))
        dirty = true  # blockmat changed — recompute before next escort
    end

    return moved_count
end