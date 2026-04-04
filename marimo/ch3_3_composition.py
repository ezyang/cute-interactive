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
    # 3.3 Composition

    Given layouts $A$ and $B$, the group composition layout $R = A \circ B$ satisfies:

    **Domain compatibility:**
    $$B \preceq R$$

    **Functional composition:**
    $$\forall c \in \mathbb{Z}(B),\; R(c) = A(B(c))$$

    In this formulation, $B$ determines the shape and coordinate sets of the resulting layout by defining the domain of $R$, while $A$ determines the codomain of $R$. The compatibility condition ensures that all coordinates of $B$ can also be used as coordinates of $R$. For admissibility of both group and functional composition, the codomain of $B$ must be compatible with the domain of $A$, which typically means that the codomain of $B$ is a set of coordinates that are congruent to one of the coordinate sets in $\mathbb{Z}(A)$.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, Tile, size, cosize, rank, depth, mode, flatten, coalesce
    from tensor_layouts import compose, complement, logical_divide, logical_product
    from tensor_layouts import zipped_divide, tiled_divide, flat_divide
    from tensor_layouts.viz import draw_layout, show_layout

    return Layout, Tile, coalesce, compose, draw_layout, mode, size


@app.cell
def _(Layout, compose, size):
    # Basic composition: R = A ∘ B means R(c) = A(B(c))
    _A = Layout((4, 8), (1, 4))
    _B = Layout(6, 2)
    _R = compose(_A, _B)
    print(f'A = {_A}')
    print(f'B = {_B}')
    print(f'R = compose(A, B) = {_R}')
    print()
    for _c in range(size(_B)):
    # Verify: R(c) = A(B(c)) for all c in domain of B
        assert _R(_c) == _A(_B(_c)), f'Failed at c={_c}'
        print(f'  R({_c}) = {_R(_c)},  A(B({_c})) = A({_B(_c)}) = {_A(_B(_c))}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3.1 Composition Properties

    **Identity Layouts.** For any shape $S$, an identity layout $I_S$ satisfies

    $$\forall c \in \mathbb{Z}_S,\; I_S(c) = c.$$

    Note that $I_S$ may actually take on any shape $P$ so long as $S \preceq P$. For example, the following are all identity layouts $I_{24}$ and satisfy $L(i) = i$ for all $i \in \mathbb{Z}_{24}$:

    $$24 : 1, \quad (4, 6) : (1, 4), \quad (3, (4, 2)) : (1, (3, 12)).$$

    For a layout $B$ with codomain $\mathbb{Z}_D$, any $I_D$ serves as a group composition left identity,

    $$I_D \circ B = B.$$

    For a layout $A$ with domain $\mathbb{Z}_S$, the layout $I_S$ with shape $S$ serves as a group composition right identity,

    $$A \circ I_S = A.$$
    """)
    return


@app.cell
def _(Layout, compose, size):
    # Identity layouts I_24: all satisfy L(i) = i for i in Z_24
    I_24a = Layout(24, 1)
    I_24b = Layout((4, 6), (1, 4))
    I_24c = Layout((3, (4, 2)), (1, (3, 12)))
    print('Identity layouts I_24:')
    for name, L in [('24:1', I_24a), ('(4,6):(1,4)', I_24b), ('(3,(4,2)):(1,(3,12))', I_24c)]:
        vals = [L(_i) for _i in range(24)]
        assert vals == list(range(24)), f'{name} is not identity!'
        print(f'  {name}: L(i) = i for all i ✓')
    print()
    _B = Layout((3, 5), (2, 7))
    I_D = Layout(size(_B) * 10, 1)
    result = compose(I_D, _B)
    # Left identity: I_D ∘ B = B
    print(f'B = {_B}')
    print(f'I_D ∘ B = {result}')  # any I_D with D large enough
    for _c in range(size(_B)):
        assert result(_c) == _B(_c)
    print('Left identity verified: I_D ∘ B = B')
    print()
    _A = Layout((4, 6), (3, 12))
    I_S = Layout((4, 6), (1, 4))
    result = compose(_A, I_S)
    print(f'A = {_A}')
    print(f'A ∘ I_S = {result}')
    # Right identity: A ∘ I_S = A
    for _c in range(size(_A)):
        assert result(_c) == _A(_c)  # identity layout with shape matching A's domain
    print('Right identity verified: A ∘ I_S = A')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Associative Property.** Given layouts $A$, $B$, and $C$, and the condition

    $$\text{image}(C) \subseteq \mathbb{Z}(B) \quad \text{and} \quad \text{image}(B) \subseteq \mathbb{Z}(A),$$

    then

    $$A \circ (B \circ C) = (A \circ B) \circ C.$$

    Note that composition is still often possible when the above condition is not satisfied, but associativity may be lost. For instance,

    $$(5, 3) : (1, 7) \circ [4 : 1 \circ 2 : 5] = (5, 3) : (1, 7) \circ 2 : 5 = 2 : 7$$

    $$[(5, 3) : (1, 7) \circ 4 : 1] \circ 2 : 5 = 4 : 1 \circ 2 : 5 = 2 : 5$$

    yield different results because the range of $2 : 5$ is not contained in the domain of $4 : 1$.
    """)
    return


@app.cell
def _(Layout, compose, size):
    # Associativity holds when image conditions are met
    _A = Layout((4, 3), (2, 8))
    _B = Layout(6, 2)
    C = Layout(3, 1)
    lhs = compose(_A, compose(_B, C))
    rhs = compose(compose(_A, _B), C)  # A ∘ (B ∘ C)
    print(f'A = {_A}, B = {_B}, C = {C}')  # (A ∘ B) ∘ C
    print(f'A ∘ (B ∘ C) = {lhs}')
    print(f'(A ∘ B) ∘ C = {rhs}')
    for _c in range(size(C)):
        assert lhs(_c) == rhs(_c)
    print('Associativity holds!')
    print()
    A2 = Layout((5, 3), (1, 7))
    B2 = Layout(4, 1)
    C2 = Layout(2, 5)
    # Associativity can fail when image conditions are violated
    lhs2 = compose(A2, compose(B2, C2))
    rhs2 = compose(compose(A2, B2), C2)
    print(f'A = {A2}, B = {B2}, C = {C2}')
    print(f'A ∘ (B ∘ C) = {lhs2}')
    print(f'(A ∘ B) ∘ C = {rhs2}')  # A ∘ (B ∘ C)
    print(f'Results differ: {lhs2} vs {rhs2}')  # (A ∘ B) ∘ C
    print('Associativity fails because image(C) ⊄ Z(B): range of 2:5 is {{0,5}} but B=4:1 has domain Z_4.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3.2 Evaluation and Restrictions

    The evaluation of group composition can be constructively derived from the layout evaluation operations. We determine a set of admissibility conditions on layouts $A$ and $B$ that are required to successfully compute group composition.

    **Base Case.** Let $B = s : d$ with $s \in \mathbb{Z}_+$ and $d \in \mathbb{N}$. Let $A = S : D = (S_0, S_1, \ldots, S_R) : (D_0, D_1, \ldots, D_R)$. Define the exclusive prefix-product of $S$ as $\overline{S}_r = \prod_{k=0}^{r-1} S_k$. The result $R = S' : D'$ has:

    $$S'_r = \frac{S_r}{\delta_r}, \quad D'_r = D_r \cdot \delta_r$$

    where $\delta_r = \lceil d / \overline{S}_r \rceil$ and $\rho_r = \lceil \overline{S}_r / d \rceil$.

    This requires two divisibility conditions:

    **Stride divisibility condition:** $\overline{S}_r \mid d$ or $d \mid \overline{S}_r$ for each $r$.

    **Shape divisibility condition:** $\lceil \overline{S}_r / d \rceil \mid s$ for each $r$.

    **Reductive Case: Distributive.** To express more general compositions in terms of the base case, we use a distributive property of the composition operation over concatenation of sublayouts of $B$:

    $$A \circ B = A \circ (B_0, B_1, \ldots) = (A \circ B_0, A \circ B_1, \ldots)$$
    """)
    return


@app.cell
def _(Layout, compose, size):
    # Base case: B = s:d (rank-1), A is multi-rank
    # The result S'_r = S_r / delta_r, D'_r = D_r * delta_r
    _A = Layout((4, 6), (1, 4))
    _B = Layout(6, 4)  # every 4th element of A, keep 6
    _R = compose(_A, _B)
    print(f'A = {_A}  (4x6 col-major, identity layout I_24)')
    print(f'B = {_B}  (every 4th element, keep 6)')
    print(f'R = A ∘ B = {_R}')
    print()
    print('Verification:')
    for _i in range(size(_B)):
        print(f'  R({_i}) = {_R(_i)},  A(B({_i})) = A({_B(_i)}) = {_A(_B(_i))}')
        assert _R(_i) == _A(_B(_i))
    return


@app.cell
def _(Layout, compose, mode, size):
    # Distributive property: A ∘ (B0, B1, ...) = (A ∘ B0, A ∘ B1, ...)
    _A = Layout((4, 8), (1, 4))
    _B = Layout((4, 3), (2, 8))
    _R = compose(_A, _B)
    R0 = compose(_A, mode(_B, 0))
    # Compose A with each mode of B separately
    _R1 = compose(_A, mode(_B, 1))
    print(f'A = {_A}')
    print(f'B = {_B}')
    print(f'A ∘ B = {_R}')
    print(f'A ∘ B0 = {R0}, A ∘ B1 = {_R1}')
    print()
    for _i in range(size(mode(_B, 0))):
        for _j in range(size(mode(_B, 1))):
            assert _R(_i, _j) == _A(_B(_i, _j)), f'Failed at ({_i},{_j})'
    # Verify pointwise
    print('Distributive property verified: compose distributes over modes of B.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3.3 Intuition and Divisibility

    Compositions with rank-1 left-hand layouts $A$ are trivial because $\overline{S}_R$ does not appear:

    $$(S_0) : (D_0) \circ s : d = s : D_0 \cdot d.$$

    In this case, there are no non-trivial divisibility checks. Note that this means group composition is still possible even when $\text{image}(B) \not\subseteq \mathbb{Z}(A)$:

    $$7 : 11 \circ 3 : 4 = 3 : 44,$$

    and $B$ does not need to be mutually disjoint in order to be distributive:

    $$7 : 11 \circ (3, 5) : (6, 3) = (3, 5) : (66, 33).$$
    """)
    return


@app.cell
def _(Layout, compose):
    # Rank-1 A: trivially composes, just scales the stride
    _A = Layout(7, 11)
    _R1 = compose(_A, Layout(3, 4))
    print(f'7:11 ∘ 3:4 = {_R1}')
    assert all((_R1(_i) == 11 * (4 * _i) for _i in range(3)))  # 3:44
    _R2 = compose(_A, Layout((3, 5), (6, 3)))
    print(f'7:11 ∘ (3,5):(6,3) = {_R2}')
    for _i in range(3):
        for _j in range(5):  # (3,5):(66,33)
            assert _R2(_i, _j) == _A(Layout((3, 5), (6, 3))(_i, _j))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For compositions with layouts $A$ of greater rank, the intuitive strategy involves two steps:

    - Determine an intermediate layout that produces every $d$th element of $A$ by "dividing out" the first $d$ elements from $A$.
    - Fix the size of the intermediate strided layout to $s$ by "keeping" the first $s$ elements.

    For example,

    $$(4, 6, 8, 10) : (2, 3, 5, 7) \circ 6 : 12$$

    is equivalent to

    $$(4, 6, 8, 10) : (2, 3, 5, 7) \circ 6 : 12 = (1, 2, 8, 10) : (X, 9, 5, 7) \circ 6 : 1,$$

    where the first 12 elements of $A$ are "divided out," and the strides are scaled accordingly. Then, the first 6 elements of the modified layout are "kept," resulting in:

    $$(1, 2, 8, 10) : (X, 9, 5, 7) \circ 6 : 1 = (2, 3) : (9, 5).$$

    Alternatively, we can "keep" the first $6 \cdot 12$ elements and then "divide out" the first 12, as follows:

    $$(4, 6, 8, 10) : (2, 3, 5, 7) \circ 6 : 12 = (4, 6, 3) : (2, 3, 5) \circ 6 : 12 = (2, 3) : (9, 5).$$
    """)
    return


@app.cell
def _(Layout, compose, size):
    # The "divide out" and "keep" intuition
    _A = Layout((4, 6, 8, 10), (2, 3, 5, 7))
    _B = Layout(6, 12)
    _R = compose(_A, _B)
    print(f'A = {_A}')
    print(f'B = {_B}')
    print(f'A ∘ B = {_R}')
    print()
    for _i in range(size(_B)):
    # Verify
        assert _R(_i) == _A(_B(_i)), f'Failed at i={_i}'
        print(f'  R({_i}) = {_R(_i)},  A({_B(_i)}) = {_A(_B(_i))}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Violations of Divisibility Conditions.** Certain compositions violate the stride or shape divisibility conditions. For example,

    $$(4, 6, 8) : (2, 3, 5) \circ 6 : 3$$

    violates the stride divisibility condition because $\overline{S}_1 = 4$ and $d = 3$ are not divisible. There is no layout that can represent every third element of the layout $(4, 6, 8) : (2, 3, 5)$.

    Similarly, the composition

    $$(4, 6, 8) : (2, 3, 5) \circ 6 : 1$$

    violates the shape divisibility condition, since $\lceil \overline{S}_1 / d \rceil = \lceil 4/1 \rceil = 4$ does not divide $s = 6$. There is no layout that can represent the first 6 elements of $(4, 6, 8) : (2, 3, 5)$.

    Ultimately, this means CuTe layouts are not strictly closed under group composition. However, in practice, violations of divisibility conditions are often due to conceptual application errors, layout/hardware incompatibilities, programmer error, and other such issues.
    """)
    return


@app.cell
def _(Layout, compose):
    # Divisibility violations

    # Stride divisibility violation: S1_bar = 4, d = 3, neither divides the other
    try:
        bad1 = compose(Layout((4, 6, 8), (2, 3, 5)), Layout(6, 3))
        print(f"(4,6,8):(2,3,5) ∘ 6:3 = {bad1} (library may handle gracefully)")
    except Exception as ex:
        print(f"Stride divisibility violation: {ex}")

    # Shape divisibility violation: ceil(4/1) = 4 does not divide s = 6
    try:
        bad2 = compose(Layout((4, 6, 8), (2, 3, 5)), Layout(6, 1))
        print(f"(4,6,8):(2,3,5) ∘ 6:1 = {bad2} (library may handle gracefully)")
    except Exception as ex:
        print(f"Shape divisibility violation: {ex}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Apparent Violations.** Consider the composition

    $$(4, 2, 8) : (3, 12, 97) \circ 3 : 3,$$

    which seemingly violates the stride divisibility condition because $\overline{S}_1 = 4$ and $d = 3$ are not divisible. This issue can be resolved by coalescing $A$ and truncating $A$. The equivalent composition produces the result

    $$(4, 2, 8) : (3, 12, 97) \circ 3 : 3 = (8, 8) : (3, 97) \circ 3 : 3 = (8) : (3) \circ 3 : 3 = 3 : 9.$$

    However, the following compositions fail divisibility conditions:

    $$(4, 2, 8) : (3, 12, 97) \circ 4 : 3$$
    $$(4, 2, 8) : (3, 15, 97) \circ 3 : 3$$

    The example on the left cannot be sufficiently truncated, while the example on the right cannot be coalesced.
    """)
    return


@app.cell
def _(Layout, coalesce, compose, size):
    # Apparent violation resolved by coalescing + truncation
    A_orig = Layout((4, 2, 8), (3, 12, 97))
    A_coal = coalesce(A_orig)
    print(f'Original A:  {A_orig}')
    print(f'Coalesced A: {A_coal}')
    print()
    A_trunc = Layout(8, 3)
    # The coalesced form is (8, 8):(3, 97). With B = 3:3, (s-1)*d = 2*3 = 6 < 8,
    # so only the first mode of A contributes. We truncate to just 8:3.
    print(f'Truncated A: {A_trunc}  (only first mode needed since (s-1)*d = 6 < 8)')
    _R = compose(A_trunc, Layout(3, 3))
    print(f'Truncated A ∘ 3:3 = {_R}')
    print()
    _B = Layout(3, 3)
    for _i in range(size(_B)):
        assert _R(_i) == A_orig(_B(_i)), f'Failed at i={_i}'
    # Verify via direct evaluation on the original
    print('Verified: R(i) = A(B(i)) for all i')
    print()
    try:
        compose(Layout((4, 2, 8), (3, 12, 97)), Layout(4, 3))
        print('(4,2,8):(3,12,97) ∘ 4:3 succeeded unexpectedly')
    except Exception as ex:
    # The left example fails: cannot be sufficiently truncated
        print(f'(4,2,8):(3,12,97) ∘ 4:3 fails: {type(ex).__name__}')
    A_bad = Layout((4, 2, 8), (3, 15, 97))
    print(f"coalesce({A_bad}) = {coalesce(A_bad)}  (modes don't merge)")
    try:
        compose(A_bad, Layout(3, 3))
        print('(4,2,8):(3,15,97) ∘ 3:3 succeeded unexpectedly')
    # The right example fails: cannot be coalesced (strides 3 and 15 not contiguous)
    except Exception as ex:
        print(f'(4,2,8):(3,15,97) ∘ 3:3 fails: {type(ex).__name__}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3.4 Application: Partitioning Example

    Composition lies at the heart of the CuTe layout algebra, enabling operations such as reshaping, restriding, permuting, partitioning, tiling, and extracting sublayouts. This section demonstrates the application of composition for partitioning a data tensor using an arbitrary thread-value pattern.

    Consider a layout of data that refines shape $(8, 8)$. We aim to partition this data using the thread-value pattern shown in Figure 6, which is the logical partitioning pattern of a specific Ampere Tensor Core. The thread-value partitioning pattern for this instruction can be represented by the layout

    ```
    ThrValLayoutC: ((4, 8), 2) : ((16, 1), 8),
    ```

    which acts as a map from `(thread_idx, value_idx)` to the 1D coordinate within the $8 \times 8$ matrix.

    Any $8 \times 8$ data layout can be permuted by composing it with the thread-value ThrValLayoutC. Each composition produces a layout compatible with shape $(32, 2)$ and defines the mapping between `(thread_idx, value_idx)` and data offsets.

    | Data Name | Data Layout 8x8 (A) | TV Layout 32x2 (B) | Result 32x2 (R) |
    |-----------|---------------------|--------------------|-----------------|
    | ColMajor | $(8, 8) : (1, 8)$ | $((4, 8), 2) : ((16, 1), 8)$ | $((4, 8), 2) : ((16, 1), 8)$ |
    | RowMajor | $(8, 8) : (8, 1)$ | $((4, 8), 2) : ((16, 1), 8)$ | $((4, 8), 2) : ((2, 8), 1)$ |
    | Padded | $(8, 8) : (1, 9)$ | $((4, 8), 2) : ((16, 1), 8)$ | $((4, 8), 2) : ((18, 1), 9)$ |
    """)
    return


@app.cell
def _(Layout, draw_layout, mode, size):
    # Thread-Value Layout for Ampere FP64 Tensor Core C-matrix
    # Maps (thread_idx, value_idx) -> 1D coordinate in 8x8 matrix
    tv_layout = Layout(((4, 8), 2), ((16, 1), 8))
    print(f"ThrValLayoutC: {tv_layout}")
    print(f"  Domain: ({size(mode(tv_layout, 0))}, {size(mode(tv_layout, 1))}) = (32 threads, 2 values)")
    print(f"  Codomain: Z_{{64}} (8x8 = 64 elements)")
    print()

    # Visualize the TV layout — this shows which thread and value own each element
    print("Thread-Value partitioning pattern (TV layout):")
    draw_layout(tv_layout, colorize=True)
    return (tv_layout,)


@app.cell
def _(Layout, compose, mode, size, tv_layout):
    # Compose various 8x8 data layouts with the TV layout
    data_col = Layout((8, 8), (1, 8))
    # Column-major data
    R_col = compose(data_col, tv_layout)
    print(f'ColMajor:  {data_col}')
    print(f'  Result:  {R_col}')
    for t in range(size(mode(tv_layout, 0))):
        for v in range(size(mode(tv_layout, 1))):
            assert R_col(t, v) == data_col(tv_layout(t, v))
    print('  Verified!')
    print()
    data_row = Layout((8, 8), (8, 1))
    _R_row = compose(data_row, tv_layout)
    # Row-major data
    print(f'RowMajor:  {data_row}')
    print(f'  Result:  {_R_row}')
    for t in range(size(mode(tv_layout, 0))):
        for v in range(size(mode(tv_layout, 1))):
            assert _R_row(t, v) == data_row(tv_layout(t, v))
    print('  Verified!')
    print()
    data_pad = Layout((8, 8), (1, 9))
    R_pad = compose(data_pad, tv_layout)
    print(f'Padded:    {data_pad}')
    # Padded data (stride 9 instead of 8 for column)
    print(f'  Result:  {R_pad}')
    for t in range(size(mode(tv_layout, 0))):
        for v in range(size(mode(tv_layout, 1))):
            assert R_pad(t, v) == data_pad(tv_layout(t, v))
    print('  Verified!')
    return


@app.cell
def _(Layout, compose, size, tv_layout):
    # The compose-and-slice pattern:
    # Given a data layout and a TV layout, compose them and then
    # slice by thread_id to get each thread's view of the data.

    data_layout = Layout((8, 8), (1, 8))  # column-major 8x8
    smem_tv = compose(data_layout, tv_layout)

    print(f"Data layout: {data_layout}")
    print(f"TV layout:   {tv_layout}")
    print(f"Composed:    {smem_tv}")
    print()

    # Slice by thread to get each thread's value layout
    for thr_id in [0, 1, 7, 31]:
        smem_v = smem_tv(thr_id, None)  # fix thread, keep value dimension
        offsets = [smem_v(v) for v in range(size(smem_v))]
        print(f"  Thread {thr_id:2d}: layout = {smem_v}, offsets = {offsets}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The code above demonstrates a common pattern that uses thread-value partitioning:

    ```python
    smem_data = Tensor(MyAccessor, MyLayout8x8)        # Tensor:    Coord -> Offset
    tv_layout = Layout(((4,8),2), ((16,1),8))          # TV Layout: (Thr,Val) -> Coord
    smem_tv   = composition(smem_data, tv_layout)      # Compose:   (Thr,Val) -> Offset
    smem_v    = smem_tv[thr_id, None]                  # Slice by thread to get subtensor
    copy(smem_v, rmem_data)                            # Copy to register tensor/array
    ```

    This pattern occurs extremely often in SIMD programming, where each processing element receives a symmetric partition of some parent data. In general, CuTe recognizes that arbitrary partitioning can be defined as composition (permutation and/or reshaping) followed by slicing. Since the partitioning pattern is very often compile-time metadata related to instructions or optimization parameters and the slice is very often a runtime program identifier index, this pattern is much more capable of propagating static information and reducing runtime overhead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3.5 By-mode Composition and Tilers

    Group composition can be applied by-mode, enabling operations on subdomains of layouts. For instance, it is often desirable to apply composition on the rows of a matrix and the columns of a matrix independently.

    **Definition 3.1.** A *Tiler* is an HTuple(Tile), where each Tile is
    - a Layout $S : D$, or
    - an Integer $S$, treated equivalently to the layout $S : 1$.

    This definition means that all layouts are tilers, all integers are tilers, and it provides for shapes like $(4, 8)$ to be used as tilers.

    Additionally, there is an equivalence between composition of tilers and composition of coordinate layouts. Specifically, all of the following objects should be considered equivalent when used in composition:

    $$(4, 8) \equiv \langle 4, 8 \rangle \equiv (4 : 1, 8 : 1) \equiv (4, 8) : (e_0, e_1)$$
    """)
    return


@app.cell
def _(Layout, Tile, compose, draw_layout, size):
    L_1 = Layout((8, 16), (1, 8))
    print(f'L = {L_1}  (8x16 column-major)')
    print(f'  size = {size(L_1)}')
    print()
    T1 = Tile(Layout(4, 1), Layout(8, 1))
    _R1 = compose(L_1, T1)
    print(f'Tiler <4:1, 8:1> (first 4 rows, first 8 cols):')
    print(f'  L ∘ T = {_R1}')
    print(f'  size = {size(_R1)}')
    draw_layout(_R1, colorize=True)
    return


@app.cell
def _(Layout, Tile, compose, draw_layout):
    L_2 = Layout((8, 16), (1, 8))
    print(f'L = {L_2}  (8x16 column-major)')
    print()
    T2 = Tile(Layout((2, 2), (1, 4)), Layout(8, 1))
    _R2 = compose(L_2, T2)
    print(f'Tiler <(2,2):(1,4), 8:1>:')
    print(f'  L ∘ T = {_R2}')
    draw_layout(_R2, colorize=True)
    return (L_2,)


@app.cell
def _(L_2, Layout, Tile, compose, draw_layout):
    # Tiler <(2,2):(1,4), 8:2> — extract rows {0,1,4,5} and every-other column
    T3 = Tile(Layout((2, 2), (1, 4)), Layout(8, 2))
    R3 = compose(L_2, T3)
    print(f'Tiler <(2,2):(1,4), 8:2>:')
    print(f'  L ∘ T = {R3}')
    draw_layout(R3, colorize=True)
    return


@app.cell
def _(L_2, Layout, Tile, compose, draw_layout):
    # Tiler <4:2, 8:2> — every-other row and every-other column
    T4 = Tile(Layout(4, 2), Layout(8, 2))
    R4 = compose(L_2, T4)
    print(f'Tiler <4:2, 8:2>:')
    print(f'  L ∘ T = {R4}')
    draw_layout(R4, colorize=True)
    return


@app.cell
def _(Layout, Tile, compose, draw_layout):
    # Verify that tilers work the same on different data layouts
    # Apply the same tiler to row-major data
    L_row = Layout((8, 16), (16, 1))
    print(f'L_row = {L_row}  (8x16 row-major)')
    T = Tile(Layout(4, 1), Layout(8, 1))
    _R_row = compose(L_row, T)
    print(f'Tiler <4:1, 8:1> on row-major:')
    print(f'  L_row ∘ T = {_R_row}')
    print()
    for _i in range(4):
        for _j in range(8):
    # Verify: R(i,j) = L(i,j) for the first 4 rows and 8 cols
    # The tiler <4:1, 8:1> selects the first 4 elements of mode 0
    # and first 8 elements of mode 1
            assert _R_row(_i, _j) == L_row(_i, _j)
    print('Verified: tiler correctly extracts 4x8 submatrix from row-major 8x16.')
    draw_layout(_R_row, colorize=True)
    return


if __name__ == "__main__":
    app.run()
