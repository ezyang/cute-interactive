# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.21.1",
#     "matplotlib==3.10.8",
#     "tensor-layouts==0.1.1",
# ]
# ///

import marimo

__generated_with = "0.22.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1.2 Canonical Loops and Loop Transformations

    The explicit calculation of loop indices is a common challenge in the development of high-performance linear algebra kernels. These calculations are difficult for programmers to get right and even more challenging to maintain. Rather than coupling information about data access with algorithmic logic, we prefer to write algorithmic logic clearly in terms of matrix/vector coordinates and abstract the data access patterns to the data layouts.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, size, idx2crd
    from tensor_layouts.viz import show_layout

    # e is an opaque pure expression — we treat it as a black box
    class E:
        """Symbolic value representing e(args). Supports equality by arguments."""
        def __init__(self, *args):
            self.args = args
        def __repr__(self):
            return f"e({', '.join(map(str, self.args))})"
        def __eq__(self, other):
            return isinstance(other, E) and self.args == other.args
        def __hash__(self):
            return hash(self.args)

    def e(*args):
        return E(*args)

    return Layout, e, idx2crd, size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To illustrate, we first define the class of loop nests that can be addressed by the techniques developed in this work. Specifically, we consider a *standard loop form* to be a loop with a single index, starting at zero, bounded by a constant, and incremented by 1 each iteration.

    For instance, consider the following loop:

    ```c
    for (int m = 2; m <= 50; m += 3)
        A[m] = e(m);
    ```

    This loop sets `A[2], A[5], A[8], ...` to the result of a pure expression `e(m)`. It can be transformed into a canonical loop as follows:

    ```c
    for (int i = 0; i < 17; ++i)
        (A + 2)[3*i] = g(i);
    ```

    Here, the pointer is offset by a loop-invariant constant, the loop stride is normalized to 1, the lower bound is transformed to zero, the upper bound is tight and non-inclusive, and the pure expression is transformed `g(i) = e(3*i+2)`.

    It is now intuitive to interpret the above example as iterating through a logically 17-element vector, where the logical coordinate is strided by 3 to index the data at base address `A + 2`. This program can be represented with the following data:

    ```
    Accessor: A + 2
    Shape:    17
    Stride:   3
    ```
    """)
    return


@app.cell
def _(e):
    # Original loop
    A_orig = {}
    for _m in range(2, 51, 3):
        A_orig[_m] = e(_m)
    A_canonical = {}
    # Canonical loop: g(i) = e(3*i + 2)
    g = lambda i: e(3 * i + 2)
    for _i in range(17):
        A_canonical[2 + 3 * _i] = g(_i)
    assert A_orig == A_canonical
    print("Original:  ", {k: v for k, v in list(A_orig.items())[:6]}, "...")
    print("Canonical: ", {k: v for k, v in list(A_canonical.items())[:6]}, "...")
    return A_orig, g


@app.cell
def _(A_orig, Layout, g, size):
    # Now with a Layout: Shape 17, Stride 3, Accessor offset 2
    layout_1d = Layout(17, 3)
    A_layout = {}
    for _i in range(size(layout_1d)):
        A_layout[2 + layout_1d(_i)] = g(_i)
    assert A_orig == A_layout
    print(f'Layout: {layout_1d}')
    print(f'Accessed indices: {[2 + layout_1d(_i) for _i in range(size(layout_1d))]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nested loops can be treated similarly. Consider the following two-dimensional loop nest:

    ```c
    for (int n = 3; n < 43; n += 2)
        for (int m = 4; m <= 22; m += 5)
            A[p*m + q*n] = e(m, n);
    ```

    which can be transformed into canonical loop form:

    ```c
    for (int j = 0; j < 20; ++j)
        for (int i = 0; i < 4; ++i)
            (A + 4*p + 3*q)[5*p*i + 2*q*j] = g(i, j);
    ```

    With the canonical loop nest, it is natural to interpret the transformed loop as iterating through a logically 4×20 matrix, where the logical row coordinate `i` is strided by `5p`, and the logical column coordinate `j` is strided by `2q` to index the data at base address `A + 4p + 3q`. This can be represented with the following data:

    ```
    Accessor: A + 4p + 3q
    Shape:    ( 4, 20)
    Stride:   (5p, 2q)
    ```
    """)
    return


@app.cell
def _(e):
    p, q = (3, 7)
    A_orig2 = {}
    # Original loop
    for n in range(3, 43, 2):
        for _m in range(4, 23, 5):
            A_orig2[p * _m + q * n] = e(_m, n)  # m <= 22, step 5: m = 4, 9, 14, 19
    A_canonical2 = {}
    g2 = lambda i, j: e(5 * i + 4, 2 * j + 3)
    # Canonical loop: g(i,j) = e(5*i + 4, 2*j + 3)
    base = 4 * p + 3 * q
    for _j in range(20):
        for _i in range(4):  # = 33
            A_canonical2[base + 5 * p * _i + 2 * q * _j] = g2(_i, _j)
    assert A_orig2 == A_canonical2
    print(f"p={p}, q={q}, base offset = 4p + 3q = {base}")
    print(f"Original and canonical loops agree on all {len(A_orig2)} accesses.")
    return A_orig2, base, g2, p, q


@app.cell
def _(A_orig2, Layout, base, g2, p, q, size):
    # With a Layout: Shape (4, 20), Stride (5p, 2q)
    layout_2d = Layout((4, 20), (5 * p, 2 * q))
    A_layout2 = {}
    for _j in range(20):
        for _i in range(4):
            A_layout2[base + layout_2d(_i, _j)] = g2(_i, _j)
    assert A_orig2 == A_layout2
    print(f'Layout: {layout_2d}')
    print(f'Base offset: {base}')
    print(f'All {size(layout_2d)} accesses match.')
    return (layout_2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A key observation is that the 4×20 matrix can also be interpreted as an 80-element vector with non-uniform, semi-affine striding, expressed with an equivalent canonical loop form:

    ```c
    for (int k = 0; k < 80; ++k)
        (A + 4*p + 3*q)[5*p*(k%4) + 2*q*(k/4)] = f(k);
    ```

    where `%` is modulo and `/` is integer floor-division.

    This transformation is the *colexicographical bijection*, `(i,j) = (k%4, k/4)`, between 2D coordinates `(i,j)` and 1D coordinates `k`. This bijection is equivalent to, and can be derived directly from, the shape represented previously. Thus, the shape representation can accept both 2D coordinates and 1D coordinates, providing a flexible and rank-agnostic framework for indexing data.
    """)
    return


@app.cell
def _(A_orig2, base, e, idx2crd, layout_2d, p, q):
    # The 1D loop with the colexicographic bijection
    # f(k) = g(k%4, k//4) = e(5*(k%4) + 4, 2*(k//4) + 3)
    A_1d = {}
    f = lambda k: e(5 * (k % 4) + 4, 2 * (k // 4) + 3)
    for k in range(80):
        A_1d[base + 5 * p * (k % 4) + 2 * q * (k // 4)] = f(k)
    assert A_orig2 == A_1d
    print('1D loop with colexicographic bijection matches original.')
    print()
    print('1D vs 2D indexing into the same layout:')
    for k in range(8):
    # The layout naturally handles both 1D and 2D coordinates
        _i, _j = (k % 4, k // 4)
        assert layout_2d(k) == layout_2d(_i, _j)
        print(f'  k={k}: idx2crd({k}, (4,20)) = {idx2crd(k, (4, 20))},  layout({k}) = layout({_i},{_j}) = {layout_2d(k)}')
    assert all((layout_2d(k) == layout_2d(k % 4, k // 4) for k in range(80)))
    print('\nAll 80 elements: 1D and 2D indexing agree.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Furthermore, the canonical loop form also provides guidance for provably correct loop transformations that often appear in optimizing compilers for tensor computations. Consider the most general canonical loop nest:

    ```c
    for (int i0 = 0; i0 < N0; ++i0)
        for (int i1 = 0; i1 < N1; ++i1)
            for (int i2 = 0; i2 < N2; ++i2)
                ...
                A[d0*i0 + d1*i1 + d2*i2 + ...] = e(i0, i1, i2, ...);
    ```

    where `(N0, N1, N2, ...)` is the "shape" of the computation and `(d0, d1, d2, ...)` are the "strides" of the access pattern,

    ```
    Accessor: A
    Shape:    (N0, N1, N2, ...)
    Stride:   (d0, d1, d2, ...)
    ```

    Because there is a one-to-one correspondence between the `Shape : Stride` information and the loop nest itself, rather than asking how to perform transformations on the loops — splitting, transposition, concatenation, permutation, truncation, vectorization, etc — we can instead ask "What are valid ways to transform the `Shape : Stride` representation and what operators provide those transformations?" Indeed, if `L = Shape : Stride` represents the data access and the loop nest, what functions `P` exist such that

    $$L' = P(L) = L \circ P$$

    is a meaningful transformation of `L = Shape : Stride`, with a certain shape and stride, to a new loop nest `L' = Shape' : Stride'` with a potentially new shape and stride. These transformations, `P`, essentially rewrite the loop nest and, if defined properly, may themselves be composable, invertible, and provide functional-programming-like control of imperative loops. With considerations of the bijection between the 1D coordinates and the ND coordinates discussed above, this paper demonstrates a very effective representation of these transformation operators is `P = Shape* : Stride*`, the same objects that we use to represent the data access and loop nests themselves.
    """)
    return


if __name__ == "__main__":
    app.run()
