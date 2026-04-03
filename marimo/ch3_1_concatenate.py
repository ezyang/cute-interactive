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
    # 3 Layout Algebra

    While CUTE layouts are only a subset of the space of all possible functions, they are capable of representing a strictly larger set of layout functions than the traditional flat-shape and flat-stride representations found in libraries like BLAS, `torch.tensor`, and `numpy.ndarray`.

    Beyond their representational power, a key utility of CUTE layouts lies in their ability to be manipulated and combined to create new layouts. This is achieved through a core set of algebraic operations defined over layouts, which can be further used to construct higher-level operations.

    In this section, we define layout homomorphisms -- operations that take CUTE layout(s) and produce a CUTE layout that satisfies some functional properties.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.1 Concatenate

    A layout can be expressed as the concatenation of its sublayouts,

    $$L = S : D = (S_0, S_1, \ldots, S_n) : (D_0, D_1, \ldots, D_n) = (S_0 : D_0,\; S_1 : D_1,\; \ldots,\; S_n : D_n) = (L_0, L_1, \ldots, L_n)$$

    such that

    $$\forall\, c = (c_0, c_1, \ldots, c_n) \in \mathbb{Z}(L),\quad L(c) = L_0(c_0) + L_1(c_1) + \cdots + L_n(c_n). \tag{11}$$
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, size, cosize, rank, depth, mode, flatten, coalesce
    from tensor_layouts.viz import draw_layout, show_layout

    return Layout, coalesce, draw_layout, mode, size


@app.cell
def _(Layout, mode):
    # A rank-2 layout: its two sublayouts are its modes
    L = Layout((4, 8), (2, 8))
    L0 = mode(L, 0)  # 4:2
    L1 = mode(L, 1)  # 8:8
    print(f'L  = {L}')
    print(f'L0 = {L0}')
    print(f'L1 = {L1}')
    print()
    for _c0 in range(4):
    # Verify Eq. (11): L(c0, c1) = L0(c0) + L1(c1)
        for _c1 in range(8):
            assert L(_c0, _c1) == L0(_c0) + L1(_c1)
    print('Verified: L(c0, c1) = L0(c0) + L1(c1) for all coordinates.')
    print()
    for _c0, _c1 in [(0, 0), (1, 0), (0, 1), (2, 3), (3, 7)]:
    # Show a few examples
        print(f'  L({_c0},{_c1}) = {L(_c0, _c1):3d}  =  L0({_c0}) + L1({_c1})  =  {L0(_c0)} + {L1(_c1)}  =  {L0(_c0) + L1(_c1)}')
    return L, L0, L1


@app.cell
def _(Layout, mode, size):
    # The same property holds for hierarchical layouts
    L_hier = Layout(((2, 3), 5), ((1, 6), 2))
    L_hier_0 = mode(L_hier, 0)  # (2,3):(1,6)
    L_hier_1 = mode(L_hier, 1)  # 5:2
    print(f'L  = {L_hier}')
    print(f'L0 = {L_hier_0}')
    print(f'L1 = {L_hier_1}')
    print()
    for _c0 in range(size(L_hier_0)):
    # Verify with 1D coordinates on each mode
        for _c1 in range(size(L_hier_1)):
            assert L_hier(_c0, _c1) == L_hier_0(_c0) + L_hier_1(_c1)
    print('Verified for hierarchical layout: L(c0, c1) = L0(c0) + L1(c1)')
    return


@app.cell
def _(Layout, size):
    # Concatenation also works in reverse: given sublayouts, we can build the full layout
    _A = Layout(4, 3)
    _B = Layout(6, 12)
    AB = Layout(_A, _B)
    print(f'A = {_A}')
    print(f'B = {_B}')
    print(f'Layout(A, B) = {AB}')
    print()
    for _c0 in range(size(_A)):
    # Verify the concatenation property
        for _c1 in range(size(_B)):
            assert AB(_c0, _c1) == _A(_c0) + _B(_c1)
    print('Verified: Layout(A, B)(c0, c1) = A(c0) + B(c1)')
    return


@app.cell
def _(L, L0, L1, draw_layout):
    # Visualize: a layout and its sublayouts
    print(f"Full layout: {L}")
    draw_layout(L, colorize=True)

    print(f"\nMode 0 (rows): {L0}")
    draw_layout(L0, colorize=True)

    print(f"\nMode 1 (cols): {L1}")
    draw_layout(L1, colorize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For concatenation admissibility, the functional property (11) implies that the codomain of all sublayouts must be contained in the same integer-semimodule. For instance, any two layouts with integer strides may be concatenated, but the layouts $4 : 2$ and $3 : e_0$ cannot be concatenated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Noting that every sublayout of a layout is also a layout, it is useful to observe that any algebraic operation that manipulates a layout can also be applied to any individual sublayout. We often call these "by-mode operations," and every operation that is defined in this section (coalesce, composition, complement, logical divide, etc) can also be applied by-mode. This approach is expressed with the combinator

    $$A \star \langle B, C \rangle = (A_0, A_1) \star \langle B, C \rangle = (A_0 \star B,\; A_1 \star C), \tag{12}$$

    where the $\star$ is some operation on two layouts and the $\langle \rangle$ notation represents a tuple of layouts, distinguishing it from a concatenation of layouts.
    """)
    return


@app.cell
def _(Layout, coalesce, mode, size):
    # Demonstrate by-mode operations using coalesce as an example
    # (This previews Section 3.2, but the by-mode combinator is defined here.)
    _A = Layout((2, (1, 6)), (1, (6, 2)))
    print(f'A = {_A}')
    print(f'coalesce(A) = {coalesce(_A)}')
    print()
    A0 = mode(_A, 0)
    A1 = mode(_A, 1)
    # By-mode coalesce: apply coalesce to each mode independently
    print(f'A0 = {A0}  ->  coalesce(A0) = {coalesce(A0)}')
    print(f'A1 = {A1}  ->  coalesce(A1) = {coalesce(A1)}')
    print()
    A_bymode = Layout(coalesce(A0), coalesce(A1))
    print(f'By-mode coalesced: {A_bymode}')
    print()
    # Reconstruct from coalesced modes
    for _i in range(size(_A)):
        assert _A(_i) == coalesce(_A)(_i)
        assert _A(_i) == A_bymode(_i)
    # The full coalesce merges everything to 12:1,
    # but by-mode preserves the rank-2 structure: (2,6):(1,2)
    # Both are functionally equivalent over their respective coordinate spaces
    print('All three are functionally equivalent over 1D coordinates.')
    return


@app.cell
def _(Layout, coalesce, mode, size):
    # Another by-mode example from the paper:
    # ((4,3), 5) : ((15,1), 3) coalesces to (4, 15):(15, 1)
    # but by-mode coalesce leaves it unchanged
    _B = Layout(((4, 3), 5), ((15, 1), 3))
    print(f'B = {_B}')
    print(f'coalesce(B) = {coalesce(_B)}')
    print()
    B0, B1 = (mode(_B, 0), mode(_B, 1))
    print(f'B0 = {B0}  ->  coalesce(B0) = {coalesce(B0)}')
    print(f'B1 = {B1}  ->  coalesce(B1) = {coalesce(B1)}')
    B_bymode = Layout(coalesce(B0), coalesce(B1))
    print(f'By-mode coalesced: {B_bymode}')
    print()
    for _i in range(size(_B)):
        assert _B(_i) == coalesce(_B)(_i)
    # By-mode coalesce preserves the structure because each mode is already minimal
    print('Functionally equivalent.')
    return


if __name__ == "__main__":
    app.run()
