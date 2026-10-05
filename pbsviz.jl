
using Plots
function plot_matrix(matrix, items, escorts, IO)
    # In this version, matrix has dimensions (ncols, nrows)
    ncols, nrows = size(matrix)
    heatmap_matrix = fill(1, ncols, nrows)  # fill first dimension as columns, second as rows

    # Mark items (storing them at [col, row])
    for itemid in keys(items)
        item = items[itemid]  # Suppose item.coords = (col, row)
        c, r = item.coords
        heatmap_matrix[c, r] = 2
    end

    for escortid in keys(escorts)
        escort= escorts[escortid]
        c, r = escort.coords
        heatmap_matrix[c, r] = 3
    end

    p = heatmap(
        heatmap_matrix',  # transpose here
        color=[:lightblue, :tomato, :white],
        axis=false,
        xlims=(0.5, ncols + 0.5),
        ylims=(0.5, nrows + 0.5),
        aspect_ratio=:equal,
        legend=false,
        colorbar=false,
        size=(60*ncols, 60*nrows)
    )

    # Draw grid lines
    for col in 1:ncols+1
        plot!(p, [col - 0.5, col - 0.5], [0.5, nrows + 0.5], color=:black, lw=1)
    end
    for row in 1:nrows+1
        plot!(p, [0.5, ncols + 0.5], [row - 0.5, row - 0.5], color=:black, lw=1)
    end
    font_size = max(8, 20 - ncols ÷ 2)  # Example scaling
    # Annotate the cells
    for col in 1:ncols
        for row in 1:nrows
            annotate!(p, col, row, text(matrix[col, row], :black, :center, font(font_size)))
        end
    end

    # Draw the IO square
    if isa(IO, Tuple)
        c, r = IO
        plot!(
            p,
            [c-0.5, c+0.5, c+0.5, c-0.5, c-0.5],
            [r-0.5, r-0.5, r+0.5, r+0.5, r-0.5],
            color=:green,
            lw=5,
            fill=false
        )
    elseif isa(IO, Vector{Tuple{Int,Int}})
        for io in IO
            c, r = io
            plot!(
                p,
                [c-0.5, c+0.5, c+0.5, c-0.5, c-0.5],
                [r-0.5, r-0.5, r+0.5, r+0.5, r-0.5],
                color=:green,
                lw=5,
                fill=false
            )
        end
    end
    
    return p
end
function get_base_filename(filepath::String)
    basename = split(filepath, '\\') |> last  # Extracts the file name part
    return split(basename, '.')[1]  # Removes the extension
end
function save_plot(saveplot, matrix, items, escorts, IO, filepath::String, save_directory::String)
    if saveplot
        plt = plot_matrix(matrix, items, escorts, IO)
        base_filename = filepath#get_base_filename(filepath)
        new_filename = joinpath(save_directory, base_filename * "_state.png")    
        # Save the plot
        savefig(new_filename)

        return new_filename
    end  # Return the path of the saved file for confirmation
end

# Example usage
#=
matrix = [
    'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h';
    'i' 'j' 'k' 'l' 'm' 'n' 'o' 'p';
    'q' 'r' 's' 't' 'u' 'v' 'w' 'x';
    'y' 'z' '1' '2' '3' '4' '5' '6';
    '7' '8' '9' '0' 'A' 'B' 'C' 'D';
    'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L'
]

items   = Dict('a' => (2,3), 'e' => (2,7))
escorts = Dict('c' => (5,2), 'i' => (3,5))
io = (1,1)
plot_matrix(matrix, items, escorts, io)
=#