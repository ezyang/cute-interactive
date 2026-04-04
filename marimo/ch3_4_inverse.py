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
    # 3.4 Inverse

    Layouts may be injective, surjective, or bijective and admit right-, left-, full-, and quasi-inverses. When layouts are interpreted as functions from coordinates to offsets, inverse layouts may be interpreted as functions from offsets to coordinates. Layout inverses are very useful in determining where within a layout certain offsets exist, extracting groups of specific offsets, or determining the common sublayout of two individual layouts.

    In this section, we define the generalized left- and right-inverses of a layout and provide application examples for their use.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, size, cosize, rank, depth, mode, flatten, coalesce
    from tensor_layouts import compose, right_inverse, left_inverse, max_common_layout
    from tensor_layouts.viz import draw_layout, show_layout

    return (
        Layout,
        compose,
        cosize,
        draw_layout,
        left_inverse,
        max_common_layout,
        right_inverse,
        size,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4.1 Right-Inverse

    A right-(pseudo)inverse of a layout $L : \mathbb{Z}_{|L|} \to D$ is an injective layout $L^\ddagger : D_{L^\ddagger} \to \mathbb{Z}_{|L|}$ that satisfies

    $$\forall k \in D_{L^\ddagger},\; L^\ddagger(L(L^\ddagger(k))) = L^\ddagger(k) \qquad (24)$$

    In the common case $D = \mathbb{Z}$, the canonical right-inverse definition is recovered,

    $$\forall k \in \mathbb{Z}_{|L^\ddagger|},\; L(L^\ddagger(k)) = k$$

    If a layout $L$ has a right-inverse layout $L^\ddagger$, then $|L^\ddagger| \le |L|$. In practice, when referencing the right-inverse of a layout, we typically mean the right-inverse with the maximum size.
    """)
    return


@app.cell
def _(Layout, right_inverse, size):
    # Table 5: Examples of layout right-inverses
    _examples = [(Layout((4, 8), (1, 4)), '(4,8):(1,4)'), (Layout((4, 8), (8, 1)), '(4,8):(8,1)'), (Layout((3, 7, 5), (5, 15, 1)), '(3,7,5):(5,15,1)'), (Layout((4, 8), (1, 5)), '(4,8):(1,5)  — non-contiguous'), (Layout((4, (4, 2)), (4, (1, 16))), '(4,(4,2)):(4,(1,16))'), (Layout(((2, 2), (4, 2)), ((1, 8), (2, 16))), '((2,2),(4,2)):((1,8),(2,16))'), (Layout(((2, 2), (2, 4)), ((0, 1), (0, 2))), '((2,2),(2,4)):((0,1),(0,2))  — stride-0')]
    for _L, _desc in _examples:
        _R = right_inverse(_L)
        print(f'L = {_desc}')
        print(f'  L‡ = {_R}  (size {size(_R)})')
        print()
    return


@app.cell
def _(Layout, right_inverse, size):
    # Verify the right-inverse property: L(L‡(k)) = k for all k in the domain of L‡
    _L = Layout((4, 8), (1, 4))
    _R = right_inverse(_L)
    print(f'L = {_L}')
    print(f'L‡ = {_R}')
    print()
    print('Verifying L(L‡(k)) = k:')
    # Check: L(L‡(k)) = k
    for _k in range(min(size(_R), 8)):
        r_k = _R(_k)
        l_r_k = _L(r_k)
        print(f'  k={_k}: L‡(k)={r_k}, L(L‡(k))={l_r_k}')
    assert all((_L(_R(_k)) == _k for _k in range(size(_R))))
    print(f'\nRight-inverse property holds for all {size(_R)} elements.')
    return


@app.cell
def _(Layout, cosize, right_inverse, size):
    # For non-contiguous layouts, the right-inverse is smaller
    L_nc = Layout((4, 8), (1, 5))
    R_nc = right_inverse(L_nc)
    print(f'L = {L_nc}  (size {size(L_nc)}, cosize {cosize(L_nc)})')
    print(f'L‡ = {R_nc}  (size {size(R_nc)})')
    print()
    print(f'Only {size(R_nc)} of {size(L_nc)} elements can be recovered — the contiguous prefix.')
    print(f'Offsets in image: {sorted(set((L_nc(_k) for _k in range(size(L_nc)))))}')
    print()
    assert all((L_nc(R_nc(_k)) == _k for _k in range(size(R_nc))))
    # Verify right-inverse still holds on its domain
    print(f'Right-inverse property holds for all {size(R_nc)} elements in domain.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When the layout $L$ is a bijection on $\mathbb{Z}_{|L|}$, then the right-inverse is also the inverse, $L^{-1}$, of the layout $L$ and satisfies

    $$\forall k \in \mathbb{Z}_{|L|},\; L(L^{-1}(k)) = k = L^{-1}(L(k))$$

    If a layout $L$ has a full-inverse $L^{-1}$ we call that layout *compact*.
    """)
    return


@app.cell
def _(Layout, right_inverse, size):
    # A compact (bijective) layout: the right-inverse is a true inverse
    _L = Layout((4, 8), (8, 1))
    _R = right_inverse(_L)
    print(f'L  = {_L}  (row-major 4x8)')
    print(f'L‡ = {_R}  (its inverse)')
    print()
    assert all((_L(_R(_k)) == _k for _k in range(size(_R))))
    # Verify BOTH directions: L(L‡(k)) = k AND L‡(L(k)) = k
    assert all((_R(_L(_k)) == _k for _k in range(size(_L))))
    print(f'Full inverse: L(L⁻¹(k))=k and L⁻¹(L(k))=k for all k in Z_{size(_L)}.')
    print(f'This layout is compact (bijective on Z_{size(_L)}).')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4.2 Application: Vectorization Example

    Right inverses are extremely useful in inspecting data layouts and determining if and where contiguous elements exist. As an immediate example, the right-inverse of the layouts $(4,8):(1,4)$ and $(4,8):(8,1)$ are $32:1$ and $(8,4):(4,1)$ respectively. This means that both layouts, because their right-inverses have size-32, index into 32 contiguous physical elements.

    As a more involved example, a common pattern for copy, called a vectorizing-copy in CuTe, attempts to find the maximum number of elements that can be copied at once between two tensors. The right-inverse allows CuTe to determine the maximum common sublayout between two layouts and, with additional information regarding hardware capabilities and physical alignment of pointers and strides, can algebraically determine the number and location of elements that can be safely vectorized to perform the copy.
    """)
    return


@app.cell
def _(Layout, draw_layout):
    # Figure 8a: A 2-element common subvector
    source_a = Layout((4, 4), (1, 4))
    dest_a   = Layout(((2, 2), 4), ((1, 8), 2))

    print("Source:", source_a)
    draw_layout(source_a, colorize=True)
    print("Destination:", dest_a)
    draw_layout(dest_a, colorize=True)
    return dest_a, source_a


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In general, for layout $A : \mathbb{Z}_{(A)} \to \mathbb{Z}_\alpha$ and $B : \mathbb{Z}_{(B)} \to \mathbb{Z}_\beta$ with $|A| = |B|$, we wish to find the largest integer $K$ such that the coordinates match

    $$\forall k \in \mathbb{Z}_K,\; A^\ddagger(k) = B^\ddagger(k)$$

    and this can be computed efficiently via finding $K$ such that

    $$\forall k \in \mathbb{Z}_K,\; k = A(B^\ddagger(k)) = (A \circ B^\ddagger)(k)$$

    or

    $$\forall k \in \mathbb{Z}_K,\; k = B(A^\ddagger(k)) = (B \circ A^\ddagger)(k)$$

    which is simply the size of the identity portion (the stride-1 mode) of $A \circ B^\ddagger = (I_K, X)$ or $B \circ A^\ddagger = (I_K, Y)$.
    """)
    return


@app.cell
def _(compose, dest_a, max_common_layout, right_inverse, size, source_a):
    # Compute the common subvector for Figure 8a
    A, _B = (source_a, dest_a)
    _B_ri = right_inverse(_B)
    comp = compose(A, _B_ri)
    print(f'A = {A}')
    print(f'B = {_B}')
    print(f'B‡ = {_B_ri}')
    print(f'A ∘ B‡ = {comp}')
    print()
    print('Values of (A ∘ B‡)(k):')
    for _k in range(size(comp)):
    # The identity prefix is the stride-1 mode at the front
        _v = comp(_k)
        _marker = '  <-- identity' if _v == _k else ''
        print(f'  k={_k}: (A ∘ B‡)({_k}) = {_v}{_marker}')
    _mcl = max_common_layout(A, _B)
    print(f'\nmax_common_layout(A, B) = {_mcl}  (size {size(_mcl)})')
    # Use max_common_layout for the direct answer
    print(f'=> {size(_mcl)} elements can be copied at once per vectorized instruction.')
    return


@app.cell
def _(Layout, draw_layout):
    # Figure 8b: A 4-element common subvector
    source_b = Layout(((2, 2), (2, 2)), ((8, 2), (4, 1)))
    dest_b   = Layout(((2, 2), (2, 2)), ((4, 2), (8, 1)))

    print("Source:", source_b)
    draw_layout(source_b, colorize=True)
    print("Destination:", dest_b)
    draw_layout(dest_b, colorize=True)
    return dest_b, source_b


@app.cell
def _(compose, dest_b, max_common_layout, right_inverse, size, source_b):
    _A2, _B2 = (source_b, dest_b)
    _B2_ri = right_inverse(_B2)
    comp2 = compose(_A2, _B2_ri)
    print(f'A = {_A2}')
    print(f'B = {_B2}')
    print(f'B‡ = {_B2_ri}')
    print(f'A ∘ B‡ = {comp2}')
    print()
    print('Values of (A ∘ B‡)(k):')
    for _k in range(size(comp2)):
        _v = comp2(_k)
        _marker = '  <-- identity' if _v == _k else ''
        print(f'  k={_k}: (A ∘ B‡)({_k}) = {_v}{_marker}')
    _mcl2 = max_common_layout(_A2, _B2)
    print(f'\nmax_common_layout(A, B) = {_mcl2}  (size {size(_mcl2)})')
    print(f'=> {size(_mcl2)} elements can be copied at once per vectorized instruction.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Moreover, the integral coordinates of these mutually contiguous elements is given by

    $$A^\ddagger \circ \lfloor B \circ A^\ddagger \rfloor_K = A^\ddagger \circ I_K = \lfloor A^\ddagger \rfloor_K : \mathbb{Z}_K \to \mathbb{Z}_{|A|}$$
    $$B^\ddagger \circ \lfloor A \circ B^\ddagger \rfloor_K = B^\ddagger \circ I_K = \lfloor B^\ddagger \rfloor_K : \mathbb{Z}_K \to \mathbb{Z}_{|B|}$$

    where $\lfloor \cdot \rfloor_K$ is a truncated-at-size-$K$ operation. This layout yields the logical coordinates of the first $K$ physical elements in the data and could be used to extract out the common subvectors from each via logical divide or zipped divide.
    """)
    return


@app.cell
def _(
    dest_a,
    dest_b,
    max_common_layout,
    right_inverse,
    size,
    source_a,
    source_b,
):
    A_1, _B = (source_a, dest_a)
    _mcl = max_common_layout(A_1, _B)
    K = size(_mcl)
    A_ri = right_inverse(A_1)
    _B_ri = right_inverse(_B)
    print(f'Figure 8a: K={K} common elements')
    print(f'  In source coords: {[A_ri(_k) for _k in range(K)]}')
    print(f'  In dest coords:   {[_B_ri(_k) for _k in range(K)]}')
    print(f'  Both map to physical offsets: {[A_1(A_ri(_k)) for _k in range(K)]}')
    print()
    _A2, _B2 = (source_b, dest_b)
    _mcl2 = max_common_layout(_A2, _B2)
    K2 = size(_mcl2)
    A2_ri = right_inverse(_A2)
    _B2_ri = right_inverse(_B2)
    print(f'Figure 8b: K={K2} common elements')
    print(f'  In source coords: {[A2_ri(_k) for _k in range(K2)]}')
    print(f'  In dest coords:   {[_B2_ri(_k) for _k in range(K2)]}')
    print(f'  Both map to physical offsets: {[_A2(A2_ri(_k)) for _k in range(K2)]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4.3 Left-Inverse

    A left-(pseudo)inverse of a layout $L : \mathbb{Z}_{|L|} \to D$ is a layout $L^\dagger : D_{L^\dagger} \to \mathbb{Z}_{|L|}$ that satisfies

    $$\forall k \in \mathbb{Z}_{|L|},\; L(L^\dagger(L(k))) = L(k) \qquad (26)$$

    The left inverse may not be unique and may take any values for inputs that are not in the image of $L$. In the common case where $L$ is injective, the canonical left-inverse definition is recovered,

    $$\forall k \in \mathbb{Z}_{|L|},\; L^\dagger(L(k)) = k.$$
    """)
    return


@app.cell
def _(Layout, left_inverse, size):
    # Table 6: Examples of layout left-inverses
    _examples = [(Layout((4, 8), (1, 4)), '(4,8):(1,4)'), (Layout((4, 8), (8, 1)), '(4,8):(8,1)'), (Layout((3, 7, 5), (5, 15, 1)), '(3,7,5):(5,15,1)'), (Layout((4, (4, 2)), (4, (1, 16))), '(4,(4,2)):(4,(1,16))'), (Layout(((2, 2), (4, 2)), ((1, 8), (2, 16))), '((2,2),(4,2)):((1,8),(2,16))')]
    for _L, _desc in _examples:
        _Li = left_inverse(_L)
        print(f'L = {_desc}')
        print(f'  L† = {_Li}  (size {size(_Li)})')
        print()
    return


@app.cell
def _(Layout, left_inverse, size):
    # Verify the left-inverse property for injective layouts: L†(L(k)) = k
    _L = Layout((4, 8), (8, 1))
    _Li = left_inverse(_L)
    print(f'L  = {_L}  (row-major 4x8)')
    print(f'L† = {_Li}')
    print()
    print('Verifying L†(L(k)) = k:')
    for _k in range(min(size(_L), 8)):
        lk = _L(_k)
        li_lk = _Li(lk)
        print(f'  k={_k}: L(k)={lk}, L†(L(k))={li_lk}')
    assert all((_Li(_L(_k)) == _k for _k in range(size(_L))))
    print(f'\nLeft-inverse property L†(L(k))=k holds for all {size(_L)} elements.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For contiguous (compact/bijective) layouts, the left- and right-inverses coincide. For non-contiguous layouts, they may differ: the right-inverse has size equal to the number of contiguous offsets from 0, while the left-inverse covers enough of the codomain to invert all values in the image.
    """)
    return


@app.cell
def _(Layout, left_inverse, right_inverse, size):
    # Compare left- and right-inverses
    layouts = [(Layout((4, 8), (1, 4)), 'bijective (col-major)'), (Layout((4, 8), (8, 1)), 'bijective (row-major)'), (Layout((3, 7, 5), (5, 15, 1)), 'bijective (3D)')]
    for _L, _desc in layouts:
        ri = right_inverse(_L)
        li = left_inverse(_L)
        print(f'L = {_L}  ({_desc})')
        print(f'  Right-inverse: {ri}  (size {size(ri)})')
        print(f'  Left-inverse:  {li}  (size {size(li)})')
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4.4 Application: Admissibility Example

    Left-inverses are useful for determining the existence and location of specific offsets produced by a data layout.

    In general, for a data layout $A : \mathbb{Z}_{(A)} \to \mathbb{Z}_\alpha$ that maps logical coordinates to data offsets, and an instruction layout $T : \mathbb{Z}_{(T)} \to \mathbb{Z}_\beta$ that maps instruction coordinates to data offsets, we wish to determine the existence and location of all offsets $T(i)$ in the image of $A$. This is stated as

    $$\forall i \in \mathbb{Z}_{|T|},\; \exists c \in \mathbb{Z}_{(A)} \;\text{s.t.}\; T(i) = A(c)$$

    and can be computed efficiently by computing the left-inverse of $A$ and checking

    $$A(A^\dagger(T(i))) = T(i)$$

    That is,
    - All offsets $T(i)$ are in the domain of $A^\dagger$.
    - All coordinates $A^\dagger(T(i))$ are unique and in the domain of $A$.

    With these conditions, we can say that each offset $T(i)$ appears in the image of $A$ and is located at the coordinate $A^\dagger(T(i))$. The layout $A^\dagger \circ T$ is a layout that maps instruction coordinates to logical data coordinates and can be used via `zipped_divide`, for example, to partition the data layout $A$ into sublayouts that correspond to the offsets accessed by the instruction.
    """)
    return


@app.cell
def _(Layout, draw_layout, left_inverse, size):
    A_2 = Layout((4, 4), (2, 8))
    A_li = left_inverse(A_2)
    print(f'Data layout A = {A_2}')
    print(f'  Image of A: {sorted(set((A_2(_k) for _k in range(size(A_2)))))}')
    print(f'  Left-inverse A† = {A_li}')
    print()
    draw_layout(A_2, colorize=True)
    return A_2, A_li


@app.cell
def _(A_2, A_li, Layout, size):
    T_good = Layout(4, 2)
    print(f'Instruction T = {T_good}')
    print(f'  Accesses offsets: {[T_good(_i) for _i in range(size(T_good))]}')
    print()
    print('Admissibility check:')
    _admissible = True
    for _i in range(size(T_good)):
        _ti = T_good(_i)
        _coord = A_li(_ti)
        _roundtrip = A_2(_coord)
        _ok = _roundtrip == _ti
        _admissible = _admissible and _ok
        print(f"  i={_i}: T(i)={_ti}, A†(T(i))={_coord}, A(A†(T(i)))={_roundtrip}  {('✓' if _ok else '✗')}")
    print(f'\nAdmissible: {_admissible}')
    print('All instruction offsets exist in the data layout.')
    return (T_good,)


@app.cell
def _(A_2, A_li, Layout, size):
    T_bad = Layout(4, 1)
    print(f'Instruction T = {T_bad}')
    print(f'  Accesses offsets: {[T_bad(_i) for _i in range(size(T_bad))]}')
    print()
    print('Admissibility check:')
    _admissible = True
    for _i in range(size(T_bad)):
        _ti = T_bad(_i)
        _coord = A_li(_ti)
        _roundtrip = A_2(_coord)
        _ok = _roundtrip == _ti
        _admissible = _admissible and _ok
        print(f"  i={_i}: T(i)={_ti}, A†(T(i))={_coord}, A(A†(T(i)))={_roundtrip}  {('✓' if _ok else '✗')}")
    print(f'\nAdmissible: {_admissible}')
    print('Odd offsets are not in the image of A, so this instruction is NOT admissible.')
    return


@app.cell
def _(A_2, A_li, T_good, compose, size):
    coord_map = compose(A_li, T_good)
    print(f'Coordinate map A† ∘ T = {coord_map}')
    print()
    print('Instruction coord -> logical data coord -> physical offset:')
    for _i in range(size(T_good)):
        logical_coord = coord_map(_i)
        physical_offset = A_2(logical_coord)
        print(f'  instr[{_i}] -> logical coord {logical_coord} -> offset {physical_offset}')
    return


if __name__ == "__main__":
    app.run()
