using Distributed
using Random
using Statistics
include("structs.jl")
include("pbsviz.jl")
include("move.jl")

function setup_workers!(n::Int)
    # no-op: parallelism now uses Threads.@threads, no worker processes needed
end

rng = MersenneTwister(1234)

function randomintialstate(matrixsize, noescorts, items, rng)
    state = Matrix{String}(undef, matrixsize[1], matrixsize[2])

    # Pre-fill with empty strings
    fill!(state, "")
    escorts = Dict{String, escort}()
    itemswithcoords = Dict{String, item}()

    # Prepare item and escort labels
    item_count = matrixsize[1] * matrixsize[2] - noescorts
    item_names = [string(i) for i in 1:item_count]
    escort_names = ["E" * string(i) for i in 1:noescorts]
    all_names = vcat(item_names, escort_names)
    
    # Shuffle the labels
    shuffle!(rng, all_names)
    
    # Fill the matrix
    idx = 1
    for j in 1:matrixsize[2], i in 1:matrixsize[1]
        currid = all_names[idx]
        state[i, j] = currid
        if currid  in escort_names
            escort = createescort(currid, (i,j), 0)
            escorts[currid] = escort
        elseif  currid in keys(items)
            item= createitem(currid, (i,j), items[currid])
            itemswithcoords[currid] =item 
        end
        idx += 1
    end
    return state, itemswithcoords, escorts
end

"""
removes items from batch when they are at IO.

mode="continue" (default, unchanged): item is delisted immediately on arrival;
its matrix cell is left as untracked debris forever (never becomes escort capacity)
— this is exactly the original behavior.

mode="leave": on arrival, the item is removed from itemstopick but kept in
`batch` for exactly one more PBSengine! round (so it still counts as a blocking
obstacle via the existing allkeys/blockmat machinery — nothing else can enter
or route through that cell). On the following call, its one-period wait is over:
it's dropped from batch and promoted into a real `escorts` entry at the same
coords, becoming usable escort capacity from then on — matching the paper's
constraint (12)/(13) one-period wait before conversion.
"""
function savemakespan_item!(makespandict,allitems, itemstopick, batch, incumbentstate, IO, time; # will need to mod for multiIO
                             mode="continue", escorts=nothing, leave_grace=nothing)
    if isa(IO, Tuple)
        io_list = [IO]
    elseif isa(IO, Vector{Tuple{Int,Int}})
        io_list = IO
    else
        throw(ArgumentError("IO in wrong format: should be a Tuple{x=int,y=int} or an Array{Tuple}"))
    end

    for (iox, ioy) in io_list
        cellid = incumbentstate[iox, ioy]

        if mode == "leave" && haskey(leave_grace, cellid)
            # one full round of idling at the IO cell elapsed -> convert to escort
            delete!(batch, cellid)
            escorts[cellid] = createescort(cellid, (iox, ioy), time)
            delete!(leave_grace, cellid)
            continue
        end

        if cellid in keys(itemstopick)
            item_obj = itemstopick[cellid]
            makespandict[cellid] = 1
            delete!(itemstopick, cellid)
            if mode == "leave"
                batch[cellid] = item_obj # ensure it's blocking as a normal batch item
                leave_grace[cellid] = time
            elseif haskey(batch, cellid)
                delete!(batch, cellid)
            end
        elseif cellid in keys(batch)
            makespandict[cellid] = time - (isa(IO, Tuple) ? 0 : allitems[cellid].tes) # allitems[itemid].tes use .tes for big batches, 0 for exact comparison
            if mode == "leave"
                leave_grace[cellid] = time # stays in batch this round, converts next round
            else
                delete!(batch, cellid)
            end
        end
    end
end
"""
pushes either the most urgent of the closest items to batch
"""
function manyincloseproxy(allcoords, IO)
    prox_items = 0
    if isa(IO, Tuple)
        iox, ioy = IO
        for coord in allcoords
            distance = abs(coord[1] - iox) + abs(coord[2] - ioy)
            if distance <= length(allcoords)
                prox_items += 1
                if prox_items >1
                    return true
                end
            end
        end
    else
        for (iox, ioy) in IO   
            for coord in allcoords
                distance = abs(coord[1] - iox) + abs(coord[2] - ioy)
                if distance <= length(allcoords)
                prox_items += 1
                    if prox_items >1
                        return true
                    end
                end
            end 
        end
    end
    
   
    return false
end
function createbatch!(batch, allitems, itemstopick, incumbentstate, time, r, IO)
    newbatch = Dict{String, item}()

    for idx in CartesianIndices(incumbentstate) # can we remove this???
        if incumbentstate[idx] in keys(itemstopick)
            item = itemstopick[incumbentstate[idx]]
            item.coords = Tuple(idx)
        end
    end

    if isa(IO, Tuple)
        iox, ioy = IO
        distances = Dict(itemid => abs(itemstopick[itemid].coords[1] - iox) + abs(itemstopick[itemid].coords[2] - ioy) for itemid in keys(itemstopick))
    elseif isa(IO, Vector{Tuple{Int,Int}})
        distances = Dict(itemid => abs(itemstopick[itemid].coords[2] - 1) for itemid in keys(itemstopick))
    else
        throw(ArgumentError("IO in wrong format: should be a Tuple{x=int,y=int} or an Array{Tuple}"))
    end
    sorted_distances = sort(collect(distances), by = x -> x[2]) 
    ebatch_coords = [item.coords for item in values(batch)]
    allcoords = vcat([item.coords for item in values(newbatch)], ebatch_coords)
    for itemid in keys(itemstopick)
        item= itemstopick[itemid]
        coords = item.coords
        l_allcoords= deepcopy(allcoords)
        push!(l_allcoords, coords)
        if item.deadline <= time && length(keys(newbatch)) < r &&
            ((manyincloseproxy(l_allcoords,IO)==false) || path_to_io_exists_if(incumbentstate, l_allcoords, IO))
            allitems[itemid].tes = time
            newbatch[itemid] = item
            push!(allcoords, coords)
            if haskey(itemstopick, itemid)
                delete!(itemstopick, itemid)
            end
            deleteat!(sorted_distances, 1)
        end
    end
    while length(newbatch) < r && !isempty(sorted_distances)
        itemid = sorted_distances[1][1]
        if !haskey(itemstopick, itemid)
            deleteat!(sorted_distances, 1)
            continue
        end
        item = itemstopick[itemid]
        coords = item.coords
        l_allcoords= deepcopy(allcoords)
        push!(l_allcoords, coords)
        if (manyincloseproxy(l_allcoords,IO)==false) || path_to_io_exists_if(incumbentstate, l_allcoords, IO)
            allitems[itemid].tes = time
            newbatch[itemid] = item
            push!(allcoords, coords)
            if haskey(itemstopick, itemid)
                delete!(itemstopick, itemid)
            end
        end
        deleteat!(sorted_distances, 1)
    end
    return newbatch
end
function changeitems!(batch, itemstopick,time,IO)
    noitems = length(keys(batch))
    closetoIO = String[]
    if isa(IO, Tuple)
        iox, ioy = IO
        for (itemid, item) in batch
            distance = abs(item.coords[1] - iox) + abs(item.coords[2] - ioy)
            if distance <= 1
                push!(closetoIO, itemid)
            end
        end
    elseif isa(IO, Vector{Tuple{Int,Int}})
        for io in IO
            iox, ioy = io
            for (itemid, item) in batch
                distance = abs(item.coords[1] - iox) + abs(item.coords[2] - ioy)
                if distance <= 1
                    push!(closetoIO, itemid)
                end
            end
        end
    else
        throw(ArgumentError("IO in wrong format: should be a Tuple{x=int,y=int} or an Array{Tuple}"))
    end
    for itemid in closetoIO
        if haskey(batch, itemid)
            item = batch[itemid]
            item.tes = 0 
            delete!(batch, itemid)
            itemstopick[itemid] = item
        end
    end
    # Randomly select items from itemstopick to add to the batch
    num_to_add = length(closetoIO)
    item_ids = collect(keys(itemstopick))
    shuffle!(rng, item_ids)
    for i in 1:min(num_to_add, length(item_ids))
        itemid = item_ids[i]
        item = itemstopick[itemid]
        itemstopick[itemid].tes = time
        batch[itemid] = item
        delete!(itemstopick, itemid)
    end
end

"""
    recalculate_makespan_by_movements(states_history, makespandict_temp)

Recalculates makespan based only on iterations where movement occurred.
For each item picked at iteration T, counts how many true movements occurred up to T.
"""
function recalculate_makespan_by_movements(states_history, makespandict_temp)
    # Build a mapping of iteration -> movement state
    movement_history = Dict{Int, Bool}()
    for (state, moved, iter, items_state, escorts_state) in states_history
        movement_history[iter] = moved
    end
    
    # For each item in makespandict_temp, recalculate based on actual movements
    recalculated_makespan = Dict{String, Int64}()
    
    for (itemid, pickup_iter) in makespandict_temp
        # Count actual time steps (movements) up to pickup_iter
        actual_time = 0
        for iter in sort(collect(keys(movement_history)))
            if iter > pickup_iter
                break
            end
            if movement_history[iter]
                actual_time += 1
            end
        end
        recalculated_makespan[itemid] = actual_time
    end
    
    return recalculated_makespan
end


function io_distance(coords, IO)
    if isa(IO, Tuple)
        return abs(coords[1] - IO[1]) + abs(coords[2] - IO[2])
    else
        return minimum(abs(coords[1] - iox) + abs(coords[2] - ioy) for (iox, ioy) in IO)
    end
end

function grasp_deadline_order(items, IO, α::Float64, rng)
    pool = collect(keys(items))
    result = String[]
    while !isempty(pool)
        dists = [io_distance(items[k].coords, IO) for k in pool]
        d_min, d_max = minimum(dists), maximum(dists)
        threshold = d_min + α * (d_max - d_min)
        rcl = [pool[i] for i in eachindex(pool) if dists[i] <= threshold]
        chosen = rcl[rand(rng, 1:length(rcl))]
        push!(result, chosen)
        filter!(k -> k != chosen, pool)
    end
    return result
end

function assign_default_deadlines!(items, matrixsize, IO, no_cores, rng)
    if isempty(items)
        return
    end
    deadlines = [it.deadline for it in values(items)]
    placeholder = first(deadlines)
    if !(placeholder == 0.0 || placeholder == 1000.0) || !all(d -> d == placeholder, deadlines)
        return
    end
    if no_cores <= 1
        for item in values(items)
            item.deadline = Float64(io_distance(item.coords, IO) +1)
        end
    else
        upper = (matrixsize[1] * matrixsize[2])^2
        order = grasp_deadline_order(items, IO, 0.1, rng)
        n = length(order)
        for (rank, itemid) in enumerate(order)
            items[itemid].deadline = Float64(ceil(Int, rank / n * upper))
        end
    end
end

"""
Stall handling, both over STALL_WINDOW[] consecutive steps (0 disables both):
- Rollback: if no target load changes position, the run is restored to the
  state just before the last step that moved a target load, and replayed with the reach-escort guard on (see reach_guard_io in move.jl)
  until the next load is picked. If every load is urgent at that point, their
  urgency is restarted (reset_urgency!). Each snapshot is rolled back to at most once.
- Forced freeroam: if no escort is assigned as a mover and no target load
  moves, idle escorts that direct serve doesn't use skip urgent serve and are
  forced into freeroam! (see force_freeroam in move.jl)
  until an escort is assigned as a mover again.
"""
const STALL_WINDOW = Ref(10)

stall_snapshot(state, batch, escorts, itemstopick, allitems, mst, leave_grace, time, nhist, stalematecheck, shuffletrigger) =
    deepcopy((state, batch, escorts, itemstopick, allitems, mst, leave_grace, time, nhist, stalematecheck, shuffletrigger))

# Restores a stall_snapshot in place; returns (time, stalematecheck, shuffletrigger).
function stall_restore!(snap, state, batch, escorts, itemstopick, allitems, mst, leave_grace, states_history)
    s = deepcopy(snap)
    state .= s[1]
    for (dst, src) in ((batch, s[2]), (escorts, s[3]), (itemstopick, s[4]), (allitems, s[5]), (mst, s[6]), (leave_grace, s[7]))
        empty!(dst)
        merge!(dst, src)
    end
    resize!(states_history, s[9])
    return s[8], s[10], s[11]
end

target_load_moved(batch, loadpos) = any(haskey(batch, k) && batch[k].coords != p for (k, p) in loadpos)

# Same urgency test as moveescorts_flow! (single IO).
function all_loads_urgent(batch, time, IO, gridsize)
    isempty(batch) && return false
    iox, ioy = IO
    diagonal_size = sqrt(gridsize[1]^2 + gridsize[2]^2)
    return all(values(batch)) do it
        floor(Int, time + (abs(iox - it.coords[1]) + it.coords[2]) * 1.5) >= it.deadline ||
        (time - it.tes) + (abs(it.coords[1] - iox) + abs(it.coords[2] - ioy)) > diagonal_size
    end
end

# Restart the loads' urgency as if they had just entered: waiting time from now,
# and a fresh Manhattan-distance deadline far enough out that none is urgent yet.
function reset_urgency!(batch, time, IO)
    for it in values(batch)
        it.tes = time
        it.deadline = time + 2 * (io_distance(it.coords, IO) + 1)
    end
end

function main(initialstate, items, escorts, IO, testid, save_directory; n=4, r=1, no_cores=1, mode="continue", deterministic_anchor=false, mm="bm", iter_cap=nothing)
    mm in ("bm", "lm") || throw(ArgumentError("mm must be \"bm\" (batch movement, default) or \"lm\" (load movement, one cell per direction per iteration), got $(repr(mm))"))
    UNIT_STEP[] = (mm == "lm")
    SINGLE_ITEM_RUN[] = (mm == "lm" && length(items) == 1)
    TOTAL_MOVES[] = 0
    ESCORT_RELOCS[] = 0
    empty!(COMMITTED_ESCORT)
    empty!(COMMITTED_IO)
    empty!(ITEM_IDLE)
    empty!(ITEM_LASTPOS)
    iteration_cap = something(iter_cap, 10 * sum(io_distance(it.coords, IO) for it in values(items); init=0))
    setup_workers!(no_cores)   # spawns workers + loads code on them if not done yet
    assign_default_deadlines!(items, size(initialstate), IO, no_cores, rng)
    allitems = deepcopy(items)
    itemstopick = deepcopy(items)
    local incumbentstate = deepcopy(initialstate)
    makespandict_temp = Dict{String, Int64}()  # Temporary, for tracking item pickup iterations
    makespandict = Dict{String, Int64}()  # Final, will be computed based on actual movements
    time = 1
    if isa(IO, Tuple)
        for item in values(allitems)
            item.assigned_io = IO
        end
    elseif isa(IO, Vector{Any})
        IO = Vector{Tuple{Int,Int}}(IO)
    end
    batch = Dict{String, item}()
    local batch = createbatch!(batch, allitems,itemstopick, incumbentstate, time, n, IO)
    leave_grace = Dict{String, Int}()  # only used when mode=="leave"
    stalematecheck = true
    shuffletrigger= false

    # Array to store states with movement info: (state, moved, iteration, items_state, escorts_state)
    states_history = Tuple{Matrix{String}, Bool, Int, Dict{String,item}, Dict{String,escort}}[]

    set_reach_guard!(nothing)
    set_force_freeroam!(false)
    stall_count = 0                 # consecutive steps without any target load moving
    noalign_count = 0               # consecutive steps with no mover assigned and no target load moving
    rollback_point = nothing        # snapshot taken just before the last step that moved a target load
    rolled_back_to = Set{Int}()     # snapshot times already rolled back to
    picks_at_guard = 0

    while !(isempty(itemstopick)&&isempty(batch))
        pre_step = STALL_WINDOW[] > 0 ? stall_snapshot(incumbentstate, batch, escorts, itemstopick, allitems, makespandict_temp,
                                                       leave_grace, time, length(states_history), stalematecheck, shuffletrigger) : nothing
        savemakespan_item!(makespandict_temp, allitems, itemstopick, batch, incumbentstate, IO, time; # deletes items from batch
                            mode=mode, escorts=escorts, leave_grace=leave_grace)
        if reach_guard_io() !== nothing && length(makespandict_temp) > picks_at_guard
            set_reach_guard!(nothing)   # a load was picked since the rollback: guard off again
        end
        # In leave mode, an item that reached IO stays in `batch` for one grace
        # round (see savemakespan_item!) purely so it keeps blocking — it's not
        # a real active target any more. Don't count it against the batch-size
        # threshold, so a new item can be activated during that same round
        # instead of waiting a full extra iteration for the grace item to
        # convert.
        grace_count = mode == "leave" ? length(leave_grace) : 0
        if length(batch) - grace_count <= n-r #decide on batch
            newcandidates = createbatch!(batch,allitems,itemstopick, incumbentstate, time, r, IO)
            if !isempty(keys(newcandidates))
                for (key, value) in newcandidates
                    if !haskey(batch, key)
                        batch[key] = value
                    end
                end
            end
            if isempty(batch)
                break
            end
        end 
        if shuffletrigger
            changeitems!(batch,itemstopick, time,  IO)
        end
        
        #assign escorts for items unique
        loadpos = Dict(k => v.coords for (k, v) in batch)
        moved = PBSengine!(time, incumbentstate, batch, escorts, IO, obj="flowtime", no_cores=no_cores, deterministic_anchor=deterministic_anchor)
        #moved = PBSengine!(time, incumbentstate, batch, escorts, IO, obj="makespan")
        if target_load_moved(batch, loadpos)
            rollback_point = pre_step
            stall_count = 0
        else
            stall_count += 1
        end
        noalign_count = (last_n_movers() == 0 && !target_load_moved(batch, loadpos)) ? noalign_count + 1 : 0
        if STALL_WINDOW[] > 0 && noalign_count >= STALL_WINDOW[]
            set_force_freeroam!(true)
        end

        # Store state with movement info and current items/escorts state
        push!(states_history, (deepcopy(incumbentstate), moved, time, deepcopy(batch), deepcopy(escorts)))

        time +=1
        if stalematecheck
            if !moved
                shuffletrigger = true
            else
                stalematecheck = false
            end
        end
        if STALL_WINDOW[] > 0 && stall_count >= STALL_WINDOW[] && rollback_point !== nothing && !(rollback_point[8] in rolled_back_to)
            push!(rolled_back_to, rollback_point[8])
            time, stalematecheck, shuffletrigger = stall_restore!(rollback_point, incumbentstate, batch, escorts, itemstopick,
                                                                   allitems, makespandict_temp, leave_grace, states_history)
            set_reach_guard!(IO)
            if IO isa Tuple && all_loads_urgent(batch, time, IO, size(incumbentstate))
                reset_urgency!(batch, time, IO)
            end
            picks_at_guard = length(makespandict_temp)
            rollback_point = nothing
            stall_count = 0
            noalign_count = 0
            set_force_freeroam!(false)
            continue
        end
        if time > iteration_cap
            break
        end
    end
    set_reach_guard!(nothing)
    set_force_freeroam!(false)

    # Post-process: save all plots
    for (state, moved, iter, items_state, escorts_state) in states_history
        save_plot(saveplot, state, items_state, escorts_state, IO, "$(testid)_$(iter)_test", save_directory)
    end
    if time > 0.9 * iteration_cap
        println("Warning: reached iteration limit without completing all items. Check for potential issues.")
    end
    # Post-process: calculate makespan based only on actual movements
    makespandict = recalculate_makespan_by_movements(states_history, makespandict_temp)
    
    return incumbentstate, makespandict, time-1
end
function main_savenow(initialstate, items, escorts, IO, testid, save_directory; n=4, r=1, no_cores=1, mode="continue")
    setup_workers!(no_cores)
    assign_default_deadlines!(items, size(initialstate), IO, no_cores, rng)
    allitems = deepcopy(items)
    itemstopick = deepcopy(items)
    local incumbentstate = deepcopy(initialstate)
    makespandict_temp = Dict{String, Int64}()
    makespandict = Dict{String, Int64}()
    time = 1
    if isa(IO, Tuple)
        for item in values(allitems)
            item.assigned_io = IO
        end
    elseif isa(IO, Vector{Any})
        IO = Vector{Tuple{Int,Int}}(IO)
    end
    batch = Dict{String, item}()
    local batch = createbatch!(batch, allitems, itemstopick, incumbentstate, time, n, IO)
    leave_grace = Dict{String, Int}()  # only used when mode=="leave"
    stalematecheck = true
    shuffletrigger = false

    states_history = Tuple{Matrix{String}, Bool, Int, Dict{String,item}, Dict{String,escort}}[]

    while !(isempty(itemstopick) && isempty(batch))
        savemakespan_item!(makespandict_temp, allitems, itemstopick, batch, incumbentstate, IO, time;
                            mode=mode, escorts=escorts, leave_grace=leave_grace)
        if length(batch) <= n - r
            newcandidates = createbatch!(batch, allitems, itemstopick, incumbentstate, time, r, IO)
            if !isempty(keys(newcandidates))
                for (key, value) in newcandidates
                    if !haskey(batch, key)
                        batch[key] = value
                    end
                end
            end
            if isempty(batch)
                break
            end
        end
        if shuffletrigger
            changeitems!(batch, itemstopick, time, IO)
        end

        moved = PBSengine!(time, incumbentstate, batch, escorts, IO, obj="flowtime", no_cores=no_cores)

        push!(states_history, (deepcopy(incumbentstate), moved, time, deepcopy(batch), deepcopy(escorts)))

        # Save immediately after each iteration
        save_plot(saveplot, incumbentstate, batch, escorts, IO, "$(testid)_$(time)_test", save_directory)

        time += 1
        if stalematecheck
            if !moved
                shuffletrigger = true
            else
                stalematecheck = false
            end
        end
        if time > 1000
            break
        end
    end

    makespandict = recalculate_makespan_by_movements(states_history, makespandict_temp)

    return incumbentstate, makespandict, time - 1
end
#= function checksync_main(matrix, escorts, items)
    for eid in keys(escorts)
        ex, ey = escorts[eid].coords
        if matrix[ex, ey] != eid
            println("CSM Escort $eid is not in the right place")
        end
    end
    for iid in keys(items)
        ix, iy = items[iid].coords
        if matrix[ix, iy] != iid
            println("CSM Item $iid is not in the right place")
        end
    end
end
function checkmatchange_main(matrix1, matrix2)
    for idx in CartesianIndices(matrix1)
        if matrix1[idx] != matrix2[idx]
            return true # change happened
        end
    end
    return false
end =#
#=
global makespan_sum = 0
global average_time_sum = 0
num_iterations = 100
timerstart = time()
for i in 1:num_iterations
    if i ==263
        global saveplot = true
    else
        global saveplot = true
    end
    item_deadlines = Dict("$i" => Float64(1000 - i * 10) for i in 1:20) 
    #item_deadlines = Dict("$i" => Float64(20 + (i - 1) * 10) for i in 1:10)
    IO= (1,1)
    initialstate, items, escorts = randomintialstate((10, 10), 4, item_deadlines, rng)
    save_directory = raw"C:\codestuff\PBS\plots\\"
    save_plot(saveplot, initialstate, items, escorts, IO, "$(0)_test", save_directory)
    finalstate, makespandict, makespan = main(initialstate, items, escorts, IO, i, save_directory)
    sumtime = 0
    max_time = makespan
    if isempty(makespandict)
        println("Empty makespandict in iteration $i")
        print_matrix(finalstate)
        println(initialstate)
        println("Items: ", items)
        println("Escorts: ", escorts)
        continue
    end
    for itemid in keys(makespandict)
        sumtime += makespandict[itemid] 
    end
    average_time = sumtime / length(keys(makespandict))
    #println("Average time: ", average_time, "       Total makespan: ", max_time)
    global makespan_sum += max_time
    global average_time_sum += average_time
end
timerstop = time()

average_makespan = makespan_sum / num_iterations
average_of_average_time = average_time_sum / num_iterations
println("Time elapsed: ", timerstop - timerstart)
println("Average of makespan: ", average_makespan)
println("Avg of Avg times: ", average_of_average_time)
=#

