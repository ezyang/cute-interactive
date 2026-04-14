import marimo

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.22.0",
#     "matplotlib",
#     "tensor-layouts>=0.2.0",
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
    # 3.2 Coalesce

    Given a layout $A$, a coalesced layout $R$ is a layout that satisfies:

    **Consistent integral domain:**
    $$|R| = |A|, \tag{13}$$

    **Flattened or integral shape:**
    $$\text{depth}(R) \le 1, \tag{14}$$

    **Consistent integral evaluation:**
    $$\forall c \in \mathbb{Z}_{|A|},\ R(c) = A(c). \tag{15}$$

    The coalesce operation "simplifies" the layout $A$ by treating it as a function over integers and potentially collapsing its shape to a shallower representation. While this process may remove rank and hierarchical information, modify coordinate sets, and merge multiple modes of $A$, it guarantees that the layout remains functionally equivalent as a mapping over its integral coordinates.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, size, depth, mode, coalesce
    from tensor_layouts.viz import draw_layout

    return Layout, coalesce, depth, draw_layout, mode, size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In practice, when referencing a coalesced layout, we typically mean the coalesced layout that achieves minimal rank.

    As an example, the coalesced layout of $(2, (1, 6)) : (1, (6, 2))$ is $12 : 1$.
    """)
    return


@app.cell
def _(Layout, coalesce, depth, size):
    A1 = Layout((2, (1, 6)), (1, (6, 2)))
    R1 = coalesce(A1)
    print(f'A = {A1}')
    print(f'coalesce(A) = {R1}')
    print()
    assert size(R1) == size(A1), 'Consistent integral domain'
    # Verify the three properties
    assert depth(R1) <= 1, 'Flattened or integral shape'
    for _c in range(size(A1)):
        assert R1(_c) == A1(_c), f'Evaluation mismatch at c={_c}'
    print(f'|R| = |A| = {size(A1)}')
    print(f'depth(R) = {depth(R1)} <= 1')
    print('R(c) = A(c) for all c in Z_|A| ✓')
    return A1, R1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, this layout can be coalesced by-mode with Eq. (12), where we apply coalesce to each mode independently:

    $$\text{coalesce}((2, (1, 6)) : (1, (6, 2)),\ \langle *, * \rangle) = \text{coalesce}((2 : 1,\ (1, 6) : (6, 2)),\ \langle *, * \rangle)$$
    $$= (\text{coalesce}(2 : 1, *),\ \text{coalesce}((1, 6) : (6, 2), *))$$
    $$= (\text{coalesce}(2 : 1),\ \text{coalesce}((1, 6) : (6, 2)))$$
    $$= (2 : 1,\ 6 : 2)$$
    $$= (2, 6) : (1, 2).$$
    """)
    return


@app.cell
def _(A1, Layout, coalesce):
    # By-mode coalesce: apply coalesce to each mode independently
    R1_bymode = coalesce(A1, (None, None))
    print(f"\nBy-mode coalesced: {R1_bymode}")
    assert R1_bymode == Layout((2, 6), (1, 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly, the rank-2 layout $((4, 3), 5) : ((15, 1), 3)$ coalesces to $(4, 15) : (15, 1)$ and the by-mode coalesced layout is $((4, 3), 5) : ((15, 1), 3)$ because the row and column layouts remain unchanged when individually coalesced.
    """)
    return


@app.cell
def _(Layout, coalesce, mode, size):
    A2 = Layout(((4, 3), 5), ((15, 1), 3))
    R2 = coalesce(A2)
    print(f'A = {A2}')
    print(f'coalesce(A) = {R2}')
    assert R2 == Layout((4, 15), (15, 1))
    for _c in range(size(A2)):
    # Verify functional equivalence
        assert R2(_c) == A2(_c)
    print(f'Functionally equivalent over all {size(A2)} integral coordinates.')
    print()
    _m0 = mode(A2, 0)
    _m1 = mode(A2, 1)
    # By-mode coalesce: each mode is already minimal
    print(f'Mode 0: {_m0}  ->  coalesce: {coalesce(_m0)}')  # (4, 3) : (15, 1) — not contiguous, can't merge
    print(f'Mode 1: {_m1}  ->  coalesce: {coalesce(_m1)}')  # 5 : 3 — already rank-1
    print(f'By-mode coalesced: Layout({coalesce(_m0)}, {coalesce(_m1)})')
    print('Modes are unchanged — by-mode coalesce equals original layout.')
    return A2, R2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rank-2 layout $(4, (3, 5)) : (15, (1, 3))$ also coalesces to $(4, 15) : (15, 1)$ and the by-mode coalesced layout is $(4, 15) : (15, 1)$ because the second mode can be coalesced individually to $15 : 1$.
    """)
    return


@app.cell
def _(Layout, coalesce, mode, size):
    A3 = Layout((4, (3, 5)), (15, (1, 3)))
    R3 = coalesce(A3)
    print(f'A = {A3}')
    print(f'coalesce(A) = {R3}')
    assert R3 == Layout((4, 15), (15, 1))
    for _c in range(size(A3)):
    # Verify functional equivalence
        assert R3(_c) == A3(_c)
    print(f'Functionally equivalent over all {size(A3)} integral coordinates.')
    print()
    _m0 = mode(A3, 0)
    _m1 = mode(A3, 1)
    # By-mode: second mode (3, 5) : (1, 3) coalesces to 15 : 1
    print(f'Mode 0: {_m0}  ->  coalesce: {coalesce(_m0)}')
    print(f'Mode 1: {_m1}  ->  coalesce: {coalesce(_m1)}')
    R3_bymode = Layout(coalesce(_m0), coalesce(_m1))
    print(f'By-mode coalesced: {R3_bymode}')
    assert R3_bymode == Layout((4, 15), (15, 1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing coalesce

    Coalesce simplifies the layout representation without changing the underlying mapping. The logical grid looks the same — only the coordinate structure changes.
    """)
    return


@app.cell
def _(A1, R1, draw_layout):
    print("Before coalesce: (2, (1, 6)) : (1, (6, 2))")
    draw_layout(A1, colorize=True)
    print(f"\nAfter coalesce: {R1}")
    draw_layout(R1, colorize=True)
    return


@app.cell
def _(A2, R2, draw_layout):
    print(f"Before coalesce: {A2}")
    draw_layout(A2, colorize=True)
    print(f"\nAfter coalesce: {R2}")
    draw_layout(R2, colorize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### When does coalesce merge modes?

    Coalesce merges two adjacent modes when the first mode's span equals the second mode's stride — i.e., the modes are contiguous in memory. When modes are not contiguous, they cannot be merged.
    """)
    return


@app.cell
def _(Layout, coalesce):
    # Contiguous modes: stride of mode 1 = size of mode 0 * stride of mode 0
    contiguous = Layout((4, 3), (1, 4))  # 4*1 = 4 = stride of mode 1 => can merge
    print(f"Contiguous: {contiguous}  ->  coalesce: {coalesce(contiguous)}")
    assert coalesce(contiguous) == Layout(12, 1)

    # Non-contiguous: gap between modes
    gapped = Layout((4, 3), (1, 8))  # 4*1 = 4 != 8 = stride of mode 1 => cannot merge
    print(f"Gapped:     {gapped}  ->  coalesce: {coalesce(gapped)}")
    assert coalesce(gapped) == Layout((4, 3), (1, 8))  # unchanged

    # Non-contiguous: modes overlap
    overlapping = Layout((4, 3), (1, 2))  # 4*1 = 4 != 2 = stride of mode 1 => cannot merge
    print(f"Overlapping:{overlapping}  ->  coalesce: {coalesce(overlapping)}")
    assert coalesce(overlapping) == Layout((4, 3), (1, 2))  # unchanged

    # Column-major: naturally contiguous
    colmajor = Layout((3, 4, 2), (1, 3, 12))
    print(f"Col-major:  {colmajor}  ->  coalesce: {coalesce(colmajor)}")
    assert coalesce(colmajor) == Layout(24, 1)
    return


@app.cell
def _(Layout, coalesce, size):
    # Partial coalescing: only some adjacent modes merge
    partial = Layout((2, 3, 5, 4), (1, 2, 10, 50))
    # Modes 0,1 are contiguous (2*1=2), modes 2,3 are contiguous (5*10=50),
    # but modes 1,2 are not (3*2=6 != 10)
    print(f'Partial: {partial}  ->  coalesce: {coalesce(partial)}')
    R_partial = coalesce(partial)
    # Verify
    for _c in range(size(partial)):
        assert R_partial(_c) == partial(_c)
    print(f'Functionally equivalent over all {size(partial)} coordinates.')
    return


if __name__ == "__main__":
    app.run()
