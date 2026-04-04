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
    # 2.5 Tensor

    Finally, we define tensors which are the central object of CuTe by binding a layout to an accessor. The accessor is effectively a random-access, pointer-like object for the layout.

    **Definition 2.18.** An accessor is an object that supports offset and dereference operations:

    - $e + d \to e'$, offset accessor $e$ by $d \in D$ to produce another accessor $e'$;
    - $*e \to v$, dereference accessor $e$ to produce value $v$;
    - $e[d] \to *(e + d)$, subscript operator as a common convenience.

    When $D = \mathbb{Z}$, common implementations of an accessor include raw pointers (e.g., `T*`), arrays (e.g. `T[N]`), and random-access iterators (e.g. `thrust::counting_iterator` and `thrust::transform_iterator`, etc).

    **Definition 2.19.** A tensor is defined by the composition of an accessor, $e$, with a layout, $L$, expressed as $T = e \circ L$. A tensor evaluates the layout by mapping a coordinate $c \in \mathbb{Z}(L)$ to the codomain $D$, offsets the accessor accordingly, and dereferences the result to obtain the tensor's value. Formally,

    $$T(c) = (e \circ L)(c) = *(e + L(c)) = e[L(c)],$$

    yields the value of the tensor at coordinate $c$.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, Tensor, size, cosize, rank, depth, mode
    from tensor_layouts.viz import draw_layout, show_layout

    return Layout, Tensor, draw_layout, mode, size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most tensors are data layouts that use a memory address as an accessor. For instance, a memory address $p$ can be used as a pointer accessor $\{p\}$ with normal offset and dereference operations to construct a data tensor,

    $$\{p\} + b \to \{p + b\}, \quad *\{p\} \to *p, \quad T = \{p\} \circ L.$$

    In addition, all layouts can be transformed into implicit tensors by composing them with a counting iterator $\{a\}$ which dereferences to a stored offset $a \in \mathbb{Z}$,

    $$\{a\} + b \to \{a + b\}, \quad *\{a\} \to a, \quad T = \{a\} \circ L.$$

    In `tensor_layouts`, a `Tensor` wraps a `Layout` with an integer offset (acting as a counting iterator).
    """)
    return


@app.cell
def _(Layout, Tensor):
    # A data tensor: 4x8 column-major matrix at base offset 0
    L = Layout((4, 8), (1, 4))
    T = Tensor(L, offset=0)
    print(f'Tensor: {T}')
    print(f'  Layout: {T.layout}')
    print(f'  Offset: {T.offset}')
    print()
    print('Evaluating T(c) = offset + L(c):')
    # T(c) = *(e + L(c)) = e[L(c)]
    # With a counting iterator, T(c) = offset + L(c)
    for _j in range(3):
        for _i in range(4):
            val = T(_i, _j)
            assert val == T.offset + L(_i, _j)
            print(f'  T({_i},{_j}) = {T.offset} + L({_i},{_j}) = {T.offset} + {L(_i, _j)} = {val}')
        print()
    return (L,)


@app.cell
def _(L, Tensor):
    # A tensor with a nonzero offset acts like a pointer offset from base
    T_offset = Tensor(L, offset=100)
    print(f"Tensor with offset=100: {T_offset}")
    print(f"  T(0,0) = {T_offset(0,0)}")
    print(f"  T(2,3) = {T_offset(2,3)}")
    print(f"  = 100 + L(2,3) = 100 + {L(2,3)} = {100 + L(2,3)}")
    return


@app.cell
def _(L, draw_layout):
    # Visualize the layout of our tensor
    print("Layout of the 4x8 column-major tensor:")
    draw_layout(L, colorize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.5.1 Slicing

    A tensor may be either fully evaluated or partially evaluated through slicing. As a CuTe Tensor can be thought of as a Layout with an offset, arbitrary slicing can be performed along any mode(s) of a natural coordinate.

    - **Full evaluation:** Applying Eq. (9) with a complete coordinate $c$ results in a value.
    - **Partial evaluation (Slicing):** When slicing with an incomplete coordinate $c = c' + c^*$, where $c^*$ represents the unspecified portion of $c$, the result is a new tensor. The operation is expressed as:

    $$T(c) = (e \circ L)(c' + c^*) = (e + L(c')) \circ L(c^*) = e' \circ L(c^*) = T'(c^*),$$

    where $L(c')$ can be fully evaluated and accumulated into $e$ and $L(c^*)$ is the sublayout of $L$ that remains unevaluated. Slicing creates a sub-tensor that can be further evaluated or manipulated.
    """)
    return


@app.cell
def _(Layout):
    L_1 = Layout((4, 8), (1, 4))
    sub_col3 = L_1(None, 3)
    off_col3 = L_1(0, 3)
    print(f'L = {L_1}')
    print(f'Slice L(None, 3):  sublayout = {sub_col3},  offset = {off_col3}')
    print(f'  -> Tensor {{offset={off_col3}}} o {sub_col3}')
    print()
    for _i in range(4):
        _original = L_1(_i, 3)
        _sliced = off_col3 + sub_col3(_i)
        assert _original == _sliced
        print(f'  L({_i}, 3) = {_original}  ==  {off_col3} + sub({_i}) = {_sliced}')
    return (L_1,)


@app.cell
def _(L_1):
    sub_row2 = L_1(2, None)
    off_row2 = L_1(2, 0)
    print(f'Slice L(2, None):  sublayout = {sub_row2},  offset = {off_row2}')
    print(f'  -> Tensor {{offset={off_row2}}} o {sub_row2}')
    print()
    for _j in range(8):
        _original = L_1(2, _j)
        _sliced = off_row2 + sub_row2(_j)
        assert _original == _sliced
        print(f'  L(2, {_j}) = {_original:2d}  ==  {off_row2} + sub({_j}) = {_sliced}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Figure 5 illustrates examples of slicing into a $6 \times 12$ matrix to extract rows, columns, and submatrices. These use a counting-iterator accessor denoted by $\{a\}$ to indicate an offset of $a \in \mathbb{Z}$. Slicing involves accumulating into the tensor offset and determining the unevaluated portion of the layout.
    """)
    return


@app.cell
def _(Layout, draw_layout, mode, size):
    # Figure 5: Tensor A = {0} o ((3,2), ((2,3),2)) : ((4,1), ((2,15),100))
    # This is a 6x12 matrix with hierarchical shape
    A = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    print(f"A = {A}")
    print(f"Shape: {A.shape}, size = {size(A)}")
    print(f"  Mode 0 size: {size(mode(A, 0))} (rows)")
    print(f"  Mode 1 size: {size(mode(A, 1))} (cols)")
    print()
    draw_layout(A, colorize=True)
    return (A,)


@app.cell
def _(A, mode, size):
    # Slice 1: A(2, _) -> {8} o ((2,3),2) : ((2,15),100)
    # Fix row coordinate to 2, keep all column coordinates
    _sub = A(2, None)
    _off = A(2, 0)
    print(f'A(2, _):')
    print(f'  offset = {_off}')
    print(f'  sublayout = {_sub}')
    print(f'  -> Tensor {{{_off}}} o {_sub}')
    print()
    for _j in range(size(mode(A, 1))):
    # Verify all elements match
        assert A(2, _j) == _off + _sub(_j)
    print(f'  Verified: all {size(mode(A, 1))} column values match.')
    return


@app.cell
def _(A, mode, size):
    # Slice 2: A(_, 5) -> {32} o (3,2) : (4,1)
    # Fix column coordinate to 5, keep all row coordinates
    _sub = A(None, 5)
    _off = A(0, 5)
    print(f'A(_, 5):')
    print(f'  offset = {_off}')
    print(f'  sublayout = {_sub}')
    print(f'  -> Tensor {{{_off}}} o {_sub}')
    print()
    for _i in range(size(mode(A, 0))):
        assert A(_i, 5) == _off + _sub(_i)
    print(f'  Verified: all {size(mode(A, 0))} row values match.')
    return


@app.cell
def _(A, size):
    # Slice 3: A(2, ((0, _), _)) -> {8} o (3, 2) : (15, 100)
    # Fix row to 2, and fix the first sub-coordinate of mode 1 to 0
    _sub = A(2, ((0, None), None))
    _off = A(2, 0)
    print(f'A(2, ((0, _, _)):')
    print(f'  offset = {_off}')
    print(f'  sublayout = {_sub}')
    print(f'  -> Tensor {{{_off}}} o {_sub}')
    print()
    for j1 in range(3):
    # This extracts a 3x2 submatrix
        for _j2 in range(2):
            assert A(2, ((0, j1), _j2)) == _off + _sub(j1, _j2)
    print(f'  Verified: all {size(_sub)} values match.')
    return


@app.cell
def _(A, mode, size):
    # Slice 4: A((_, 1), (_, 0)) -> {1} o (3, (2, 3)) : (4, (2, 15))
    # Fix second sub-coord of mode 0 to 1, first sub-coord of mode 1 to 0
    _sub = A((None, 1), (None, 0))
    _off = A((0, 1), (0, 0))
    print(f'A((_, 1), (_, 0)):')
    print(f'  offset = {_off}')
    print(f'  sublayout = {_sub}')
    print(f'  -> Tensor {{{_off}}} o {_sub}')
    print()
    for _i in range(3):
        for _j in range(size(mode(_sub, 1))):
            assert A((_i, 1), (_j, 0)) == _off + _sub(_i, _j)
    print(f'  Verified: all {size(_sub)} values match.')
    return


@app.cell
def _(A, size):
    # Slice 5: A((_, 0), ((0, _), 1)) -> {100} o (3, 3) : (4, 15)
    # Fix second sub-coord of mode 0 to 0, first sub-coord of mode 1's
    # first sub-mode to 0, and mode 1's second sub-coord to 1
    _sub = A((None, 0), ((0, None), 1))
    _off = A((0, 0), ((0, 0), 1))
    print(f'A((_, 0), ((0, _), 1)):')
    print(f'  offset = {_off}')
    print(f'  sublayout = {_sub}')
    print(f'  -> Tensor {{{_off}}} o {_sub}')
    print()
    for _i in range(3):
        for _j in range(3):
            assert A((_i, 0), ((0, _j), 1)) == _off + _sub(_i, _j)
    print(f'  Verified: all {size(_sub)} values match.')
    return


@app.cell
def _(A, size):
    # Slice 6: A((1, _), ((_, 0), _)) -> {4} o (2, (2, 2)) : (1, (2, 100))
    # Fix first sub-coord of mode 0 to 1, second sub-coord of mode 1's
    # first sub-mode to 0
    _sub = A((1, None), ((None, 0), None))
    _off = A((1, 0), ((0, 0), 0))
    print(f'A((1, _), ((_, 0), _)):')
    print(f'  offset = {_off}')
    print(f'  sublayout = {_sub}')
    print(f'  -> Tensor {{{_off}}} o {_sub}')
    print()
    for i1 in range(2):
    # Verify: sub is rank-3 since the library flattens (2, (2, 2)) -> (2, 2, 2)
        for j0 in range(2):
            for _j2 in range(2):
                assert A((1, i1), ((j0, 0), _j2)) == _off + _sub(i1, j0, _j2)
    print(f'  Verified: all {size(_sub)} values match.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The key insight is that CuTe's hierarchical shapes allow slicing at *any sub-boundary* of the layout, not just along entire modes. This is strictly more powerful than the ranged slicing offered by libraries like NumPy or PyTorch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In many tensor libraries like `numpy.ndarray`, `torch.tensor`, and MATLAB, slicing is supported with notation similar to above: write `my_matrix[2,:]` to extract the second row of a matrix and `my_matrix[:,4]` to extract the fourth column of a matrix. These libraries also support ranged slicing, such as `my_matrix[2:4,1:3]` to extract the submatrix from the second to the fourth row and the first to the third column. CuTe does not support ranged slicing as it finds ranged slicing to be problematic for several reasons:

    - Ranged slicing can't express all of the slices demonstrated in Figure 5. The last slice example cannot be expressed with ranged slicing on only the rows and columns of a matrix.
    - Ranged slicing promotes patterns like
      ```python
      thr_data = my_data[thr_id*TILE_SIZE:(thr_id+1)*TILE_SIZE]
      ```
      to retrieve a "tile" of data local to each thread. This pattern conflates the `TILE_SIZE`, which is very often a static constant that a program wants to optimize over, with a `thr_id`, which is a fundamentally dynamic index local to each thread. Instead, CuTe prefers a two-stage permute-and-slice approach.
    - Ranged slicing can express slices that are impossible to represent with a CuTe layout.
    """)
    return


if __name__ == "__main__":
    app.run()
