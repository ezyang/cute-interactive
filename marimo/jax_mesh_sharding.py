import marimo

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.22.0",
#     "matplotlib",
#     "tensor-layouts",
# ]
# ///

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Representing JAX Device Mesh Sharding with CuTe Layouts

    JAX's `jax.make_mesh` creates a device mesh by reshaping physical devices into a logical grid.
    When the physical device topology doesn't match a simple linear order, the resulting mesh
    has a non-trivial device-ID mapping. We can represent this exactly using CuTe's `Swizzle` + `Layout`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The JAX setup

    ```python
    mesh = jax.make_mesh((4, 2), ('a', 'b'))
    y = jax.device_put(x, NamedSharding(mesh, P('a', 'b')))
    jax.debug.visualize_array_sharding(y)
    ```

    ```
    ┌──────────┬──────────┐
    │  TPU 0   │  TPU 1   │
    ├──────────┼──────────┤
    │  TPU 2   │  TPU 3   │
    ├──────────┼──────────┤
    │  TPU 6   │  TPU 7   │
    ├──────────┼──────────┤
    │  TPU 4   │  TPU 5   │
    └──────────┴──────────┘
    ```

    Notice the device IDs aren't in simple row-major order: rows 2 and 3 are swapped (6,7 before 4,5).
    This reflects the physical TPU interconnect topology.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Row-major baseline

    A plain 4x2 row-major layout maps `(i,j) -> 2i + j`, giving device IDs 0,1,2,3,4,5,6,7 in order.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, Swizzle, compose
    from tensor_layouts.viz import draw_layout

    row_major = Layout((4, 2), (2, 1))
    print(row_major)
    draw_layout(row_major, colorize=True)
    return Layout, Swizzle, compose, draw_layout


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: The topology permutation as a Swizzle

    The physical device order is `[0, 1, 2, 3, 6, 7, 4, 5]` — for devices 4-7, bit 1 is flipped.
    In binary:

    | Position | Binary | Device | Binary | XOR |
    |----------|--------|--------|--------|-----|
    | 0 | 000 | 0 | 000 | — |
    | 1 | 001 | 1 | 001 | — |
    | 2 | 010 | 2 | 010 | — |
    | 3 | 011 | 3 | 011 | — |
    | 4 | **1**00 | 6 | **1**10 | bit 2 XORed into bit 1 |
    | 5 | **1**01 | 7 | **1**11 | bit 2 XORed into bit 1 |
    | 6 | **1**10 | 4 | **1**00 | bit 2 XORed into bit 1 |
    | 7 | **1**11 | 5 | **1**01 | bit 2 XORed into bit 1 |

    This is exactly `Swizzle(bits=1, base=1, shift=1)`: XOR 1 bit starting at position `base=1` with the bit at `base+shift=2`.
    """)
    return


@app.cell
def _(Swizzle):
    sw = Swizzle(1, 1, 1)
    print('Swizzle permutation:')
    for _i in range(8):
        print(f'  {_i} -> {sw(_i)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Compose to get the mesh layout

    Composing the swizzle with the row-major layout gives us the full device mesh mapping:

    `mesh(i, j) = Swizzle(row_major(i, j))`
    """)
    return


@app.cell
def _(Layout, Swizzle, compose):
    mesh_layout = compose(Swizzle(1, 1, 1), Layout((4, 2), (2, 1)))
    print(mesh_layout)
    print()
    for _i in range(4):
        row = [f'TPU {mesh_layout(_i, j)}' for j in range(2)]
        print(f'  row {_i}: {row}')
    return (mesh_layout,)


@app.cell
def _(draw_layout, mesh_layout):
    draw_layout(mesh_layout, colorize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The visualization shows the same 4x2 grid with device IDs matching the JAX output:
    TPUs 0,1 / 2,3 / 6,7 / 4,5 — the topology-aware ordering captured entirely
    by a single `Swizzle` composed with a standard row-major layout.

    ## Step 4: Sharding an 8x4 array with contiguous local chunks

    Now consider an 8×4 array sharded across this 4×2 mesh with `P('a', 'b')`.
    Each device gets a 2×2 tile. We want the local chunks to be **contiguous** in memory,
    since hopping between devices is expensive.

    We build this in two steps:
    1. `blocked_product(local_tile, device_mesh)` — tile the local 2×2 chunk across device positions
    2. Compose `Swizzle(1, 3, 1)` on the result — the same XOR permutation, but shifted to operate on the device-ID bits (bits 2–4) of the full offset
    """)
    return


@app.cell
def _(Layout, Swizzle, compose):
    from tensor_layouts import blocked_product
    local_tile = Layout((2, 2), (2, 1))
    # Local tile: 2x2 row-major (contiguous 4 elements per device, PyTorch convention)
    device_mesh = Layout((4, 2), (2, 1))
    blocked = blocked_product(local_tile, device_mesh)
    # Device mesh: plain row-major (swizzle applied later at correct bit positions)
    print('Without swizzle:', blocked)
    full = compose(Swizzle(1, 3, 1), blocked)
    # blocked_product tiles the local chunk across device positions
    print('With swizzle:   ', full)
    print()
    for _i in range(8):
    # Swizzle(1, 3, 1): same XOR as before, but shifted up by 2 bits
    # because each device owns 4 elements (2 bits), so device ID starts at bit 2
        offsets = [f'{full(_i, j):2d}' for j in range(4)]
        devices = [f'd{full(_i, j) // 4}' for j in range(4)]
    # Print the 8x4 grid with device assignments
        print(f"  row {_i}: offsets [{', '.join(offsets)}]  devices [{', '.join(devices)}]")
    return blocked_product, full


@app.cell
def _(Layout, Swizzle, blocked_product, compose, draw_layout, full):
    # Color by device ID: map each (i,j) to its device via the swizzled mesh
    color_base = blocked_product(Layout((2, 2), (0, 0)), Layout((4, 2), (2, 1)))
    color = compose(Swizzle(1, 1, 1), color_base)

    draw_layout(full, flatten_hierarchical=False, color_layout=color, colorize=True, num_colors=8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each cell shows the hierarchical coordinates from `blocked_product`'s shape `((2, 4), (2, 2))`:

    - **row** = `(local_row, device_row)` — first component is position within the 2×2 tile, second is which device row (0–3)
    - **col** = `(local_col, device_col)` — first component is position within the tile, second is which device column (0–1)
    - **offset** = linearized physical offset (with swizzle applied)

    You can read off the device grid position from the second components and the local position within the tile from the first.
    """)
    return


if __name__ == "__main__":
    app.run()
