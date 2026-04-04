# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.22.4",
#     "matplotlib==3.10.8",
#     "tensor-layouts==0.1.1",
# ]
# ///

import marimo

__generated_with = "0.22.4"
app = marimo.App()

with app.setup:
    import marimo as mo
    from tensor_layouts import Layout, blocked_product, size, zipped_divide
    from tensor_layouts.viz import draw_layout


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Representing GPU Mesh Sharding with CuTe Layouts

    On GPUs, device IDs typically follow a simple linear order (no topology permutation).
    This makes the mesh layout a plain row-major `Layout` — no `Swizzle` needed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mesh_rows = mo.ui.slider(2, 8, step=1, value=4, label="Mesh rows", show_value=True)
    mesh_cols = mo.ui.slider(2, 4, step=1, value=2, label="Mesh cols", show_value=True)
    tile_rows = mo.ui.slider(1, 4, step=1, value=2, label="Tile rows", show_value=True)
    tile_cols = mo.ui.slider(1, 4, step=1, value=2, label="Tile cols", show_value=True)
    mo.vstack([mesh_rows, mesh_cols, tile_rows, tile_cols])
    return mesh_cols, mesh_rows, tile_cols, tile_rows


@app.cell(hide_code=True)
def _(mesh_cols, mesh_rows, tile_cols, tile_rows):
    def gpu_grid_ascii(rows, cols):
        w = max(len(f"GPU {rows * cols - 1}") + 4, 10)
        h = lambda l, m, r: l + m.join(["─" * w] * cols) + r
        lines = [h("┌", "┬", "┐")]
        for i in range(rows):
            cells = "│".join(f"GPU {i * cols + j}".center(w) for j in range(cols))
            lines.append(f"│{cells}│")
            if i < rows - 1:
                lines.append(h("├", "┼", "┤"))
        lines.append(h("└", "┴", "┘"))
        return "\n".join(lines)


    _r, _c = mesh_rows.value, mesh_cols.value
    _tr, _tc = tile_rows.value, tile_cols.value
    mo.md(f"""\
    ## The setup

    {_r * _c} GPUs arranged as a {_r}×{_c} mesh, sharding a {_r * _tr}×{_c * _tc} array with `P('a', 'b')`.
    Each GPU gets a contiguous {_tr}×{_tc} tile.

    ```
    {gpu_grid_ascii(_r, _c)}
    ```

    Device IDs are in plain row-major order — no swizzling.
    """)
    return


@app.cell(hide_code=True)
def _(mesh_cols, mesh_rows):
    _r, _c = mesh_rows.value, mesh_cols.value
    mo.md(f"""\
    ## Step 1: The device mesh

    A {_r}×{_c} row-major layout directly maps mesh coordinates to GPU IDs: `(i,j) → {_c}i + j`.
    """)
    return


@app.cell
def _(mesh_cols, mesh_rows):
    device_mesh = Layout((mesh_rows.value, mesh_cols.value), (mesh_cols.value, 1))
    print(f"Device mesh: {device_mesh}")
    for _i in range(mesh_rows.value):
        _row = [f"GPU {device_mesh(_i, _j)}" for _j in range(mesh_cols.value)]
        print(f"  row {_i}: {_row}")
    return (device_mesh,)


@app.cell
def _(device_mesh):
    draw_layout(device_mesh, colorize=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Sharding with contiguous local chunks

    Each GPU gets a row-major tile (PyTorch convention: biggest stride on the left).
    `blocked_product` tiles the local chunk across device positions, pairing corresponding
    modes: row with row, column with column.
    """)
    return


@app.cell
def _(device_mesh, tile_cols, tile_rows):
    local_tile = Layout((tile_rows.value, tile_cols.value), (tile_cols.value, 1))
    full = blocked_product(local_tile, device_mesh)
    tile_size = tile_rows.value * tile_cols.value
    return full, tile_size


@app.function
def print_sharded_grid(layout, rows, cols, tile_sz):
    for i in range(rows):
        offsets = [f"{layout(i, j):2d}" for j in range(cols)]
        devices = [f"d{layout(i, j) // tile_sz}" for j in range(cols)]
        print(
            f"  row {i}: offsets [{', '.join(offsets)}]  devices [{', '.join(devices)}]"
        )


@app.cell
def _(full, mesh_cols, mesh_rows, tile_cols, tile_rows, tile_size):
    print("Layout:", full)
    print()
    print_sharded_grid(
        full,
        tile_rows.value * mesh_rows.value,
        tile_cols.value * mesh_cols.value,
        tile_size,
    )
    return


@app.cell
def _(device_mesh, full, tile_cols, tile_rows):
    color = blocked_product(
        Layout((tile_rows.value, tile_cols.value), (0, 0)), device_mesh
    )
    draw_layout(
        full,
        flatten_hierarchical=False,
        color_layout=color,
        colorize=True,
        num_colors=size(device_mesh),
    )
    return (color,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each cell shows the hierarchical coordinates from `blocked_product`:

    - **row** = `(local_row, device_row)` — position within tile, then which device row
    - **col** = `(local_col, device_col)` — position within tile, then which device column
    - **offset** = linearized physical offset

    Without a topology swizzle, offsets are simply `device_id * tile_size + local_offset`.

    ## Step 3: Top-down — partitioning a global array with `zipped_divide`

    `blocked_product` works **bottom-up**: given a local tile and a device mesh, it builds the global layout.
    But what if we start with an existing global array and want to partition it into tiles?

    `zipped_divide(global_array, tile_shape)` splits the array into `(tile, grid)`:
    the first mode indexes within a tile, the second mode selects which tile (i.e., which device).

    The key difference: `blocked_product` **creates** a layout where each device's chunk is contiguous,
    while `zipped_divide` **preserves** the original memory layout (here, row-major).
    """)
    return


@app.cell
def _(mesh_cols, mesh_rows, tile_cols, tile_rows):
    _total_r = tile_rows.value * mesh_rows.value
    _total_c = tile_cols.value * mesh_cols.value
    global_array = Layout((_total_r, _total_c), (_total_c, 1))
    tiled = zipped_divide(global_array, Layout((tile_rows.value, tile_cols.value)))

    print("Global array:", global_array)
    print()
    print("zipped_divide result:", tiled)
    print("  Shape:", tiled.shape)
    print("  Mode 0 (tile):  shape", tiled.shape[0])
    print("  Mode 1 (grid):  shape", tiled.shape[1])
    print()

    _t_size = tile_rows.value * tile_cols.value
    for _g in range(mesh_rows.value * mesh_cols.value):
        _offsets = sorted([tiled(_t, _g) for _t in range(_t_size)])
        print(f"  grid {_g} (device {_g}): offsets {_offsets}")
    return (tiled,)


@app.cell
def _(color, device_mesh, tile_cols, tile_rows, tiled):
    color_td = zipped_divide(color, Layout((tile_rows.value, tile_cols.value)))
    draw_layout(
        tiled,
        flatten_hierarchical=False,
        color_layout=color_td,
        colorize=True,
        num_colors=size(device_mesh),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparing the two approaches

    | | `blocked_product` (bottom-up) | `zipped_divide` (top-down) |
    |---|---|---|
    | **Input** | local tile + device mesh | global array + tile shape |
    | **Memory layout** | Each device's chunk is contiguous | Preserves original row-major layout |
    | **Use case** | Designing a distributed memory layout | Partitioning an existing array across devices |

    Both produce the same logical tiling (same blocks assigned to same devices), but the physical offsets differ because the memory layouts are different.
    """)
    return


if __name__ == "__main__":
    app.run()
