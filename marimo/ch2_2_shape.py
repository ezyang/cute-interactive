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
    # 2.2 Shape

    CuTe makes the observation that a 2D shape $(M, N)$ can also be interpreted to describe 1D $MN$ elements indexed
    by an integral coordinate $i$ with $0 \le i < MN$ provided a bijection

    $$S : \mathbb{Z}_{MN} \longleftrightarrow \mathbb{Z}_M \times \mathbb{Z}_N$$

    maps between the 1D integral coordinates $i \in \mathbb{Z}_{MN}$ and the 2D natural coordinates $(m, n) \in \mathbb{Z}_M \times \mathbb{Z}_N$.

    Similarly, the 2D shape $(M, NP)$ can be interpreted as a hierarchical shape $(M, (N, P))$ indexed by natural coordinates $(m, (n, p))$ with $0 \le m < M$, $0 \le n < N$, $0 \le p < P$. A similar bijection can be made

    $$S : \mathbb{Z}_M \times \mathbb{Z}_{NP} \longleftrightarrow \mathbb{Z}_M \times (\mathbb{Z}_N \times \mathbb{Z}_P)$$

    to map between the 2D coordinates $(m, q) \in \mathbb{Z}_M \times \mathbb{Z}_{NP}$ and the natural coordinates $(m, (n, p)) \in \mathbb{Z}_M \times (\mathbb{Z}_N \times \mathbb{Z}_P)$.

    A direct consequence of hierarchical shapes and coordinates is that tensor algorithms can be written for the shapes that are most natural to them (Section 2.6) – 1D shapes for vectors in COPY, 2D shapes for matrices in GEMM, 3D shapes for tensors in batched-GEMM, etc – while still accepting hierarchically shaped tensors that are folded to be weakly congruent with the algorithm’s specification (Section 2.2.1). Tensors of data, whose shape is often represented as a flat sequence of integers, can be arbitrarily folded into shapes accepted by generic tensor algorithms. Furthermore, because each mode of a tensor is associated with a stride (Section 2.3) to index data, this folding of modes allows the representation of much more complex layouts of data beyond simple contiguous arrays in COPY or row-major and col-major matrices in BLAS GEMM (Section 2.4).

    In the following sections, we define this notion of compatibility and these relations between coordinate sets within a shape.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, size, cosize, rank, depth, mode, idx2crd
    from tensor_layouts.viz import draw_layout, show_layout

    return Layout, depth, draw_layout, idx2crd, rank, size


@app.cell
def _(idx2crd):
    # Demonstrate the 1D <-> 2D bijection for shape (M, N)
    _M, _N = (3, 4)
    print(f'Shape ({_M}, {_N}) has {_M * _N} elements.')
    print(f'\n1D integral coordinates <-> 2D natural coordinates:')
    for _i in range(_M * _N):
        _crd = idx2crd(_i, (_M, _N))
        print(f'  i={_i:2d}  <->  (m, n) = {_crd}')
    return


@app.cell
def _():
    # Hierarchical shape: (M, (N, P)) — the 2D shape (M, NP) refined
    _M, _N, P = (3, 2, 4)
    print(f'Hierarchical shape ({_M}, ({_N}, {P})) has {_M * _N * P} elements.')
    print(f'\n2D coords (m, q) <-> natural coords (m, (n, p)):')
    for q in range(_N * P):
        for _m in range(_M):
            _n, p = (q % _N, q // _N)
            print(f'  (m, q) = ({_m}, {q:2d})  <->  (m, (n, p)) = ({_m}, ({_n}, {p}))')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2.1 Coordinate Sets and Compatibility

    As previously suggested, hierarchical shapes provide for indexing by multiple coordinate systems. Here, we define the coordinate set for a specific shape and a notion of compatibility between shapes to share coordinate sets between shapes.

    **Definition 2.6.** A *coordinate set* is a set $\mathbb{Z}_N = \{0, 1, 2, \ldots, N-1\}$ of non-negative integers or a Cartesian product of coordinate sets, $\mathbb{Z}_N \times \mathbb{Z}_M = \mathbb{Z}_{(N,M)}$.

    For instance, the following are examples of coordinate sets:

    $$\mathbb{Z}_6 = \{0, 1, 2, 3, 4, 5\}$$

    $$\mathbb{Z}_3 \times \mathbb{Z}_4 = \mathbb{Z}_{(3,4)} = \{(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3)\}$$

    $$(\mathbb{Z}_2 \times \mathbb{Z}_1) \times \mathbb{Z}_3 = \mathbb{Z}_{((2,1),3)} = \{((0,0),0), ((1,0),0), ((0,0),1), ((1,0),1), ((0,0),2), ((1,0),2)\}$$
    """)
    return


@app.cell
def _():
    def coord_set(shape):
        """Generate the coordinate set Z_S for a given shape, in colexicographic order."""
        if isinstance(shape, int):
            return list(range(shape))
        else:
            # Cartesian product in colexicographic order (first mode varies fastest)
            sub_sets = [coord_set(s) for s in shape]
            result = [()]
            for ss in sub_sets:
                result = [prev + (c,) for c in ss for prev in result]
            return result

    print("Z_6 =", coord_set(6))
    print()
    print("Z_(3,4) =", coord_set((3, 4)))
    print()
    print("Z_((2,1),3) =", coord_set(((2, 1), 3)))
    return (coord_set,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A coordinate set $\mathbb{Z}_S$ is precisely the set of natural coordinates for a shape $S$. Other coordinate sets for a shape $S$ are any coordinate set for a shape that is compatible with $S$ and coarsens $S$.

    **Definition 2.7.** *Compatibility*, $\preceq$, is a partial order on the set of shapes. For shapes $P$ and $S$,

    $$P \preceq S \quad \text{iff} \quad P \in \mathbb{Z}^+ \text{ and } P = |S|, \quad \text{or} \quad P, S \in \text{Tuple and } \text{rank}(P) = \text{rank}(S) \text{ and } \forall i\ P_i \preceq S_i$$

    and we say that $P$ and $S$ are *compatible*, $P$ *coarsens* $S$, and $S$ *refines* $P$.

    Compatibility requires that the two shapes be the same size, so the integral values of the HTuples matter. For example,

    $$30 \preceq (2, 15) \preceq (2, (3, 5))$$

    and

    $$30 \preceq (6, 5) \preceq ((3, 2), 5)$$

    but $(2, (3, 5))$ and $((3, 2), 5)$ are *not* compatible despite having the same size. They do, however, share a common compatible shape of $30$.
    """)
    return


@app.cell
def _():
    def shape_size(s):
        """Compute |S| — the product of all integers in a (possibly nested) shape."""
        if isinstance(s, int):
            return s
        result = 1
        for x in s:
            result *= shape_size(x)
        return result

    def is_compatible(p, s):
        """Check if shape P coarsens shape S (P ⪯ S)."""
        if isinstance(p, int):
            return p == shape_size(s)
        if isinstance(s, int):
            return False  # tuple can't coarsen an int (unless p is int, handled above)
        if len(p) != len(s):
            return False
        return all(is_compatible(pi, si) for pi, si in zip(p, s))

    # 30 ⪯ (2, 15) ⪯ (2, (3, 5))
    print(f"30 ⪯ (2, 15):       {is_compatible(30, (2, 15))}")
    print(f"(2, 15) ⪯ (2, (3,5)): {is_compatible((2, 15), (2, (3, 5)))}")
    print(f"30 ⪯ (2, (3, 5)):   {is_compatible(30, (2, (3, 5)))}")
    print()

    # 30 ⪯ (6, 5) ⪯ ((3, 2), 5)
    print(f"30 ⪯ (6, 5):        {is_compatible(30, (6, 5))}")
    print(f"(6, 5) ⪯ ((3,2),5): {is_compatible((6, 5), ((3, 2), 5))}")
    print()

    # (2, (3, 5)) and ((3, 2), 5) are NOT compatible
    print(f"(2,(3,5)) ⪯ ((3,2),5): {is_compatible((2, (3, 5)), ((3, 2), 5))}")
    print(f"((3,2),5) ⪯ (2,(3,5)): {is_compatible(((3, 2), 5), (2, (3, 5)))}")
    print("Both have size 30, but they are not compatible.")
    return (shape_size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the definition of a coordinate set and shape compatibility, we can define the set of all compatible coordinates for any given shape.

    **Definition 2.8.** A shape $S$ defines a set of compatible coordinate sets, $\mathcal{Z}(S)$, as the coordinate sets of all shapes that coarsen $S$.

    $$\mathcal{Z}(S) = \{\mathbb{Z}_{S'} \mid S' \preceq S\}. \qquad (3)$$

    Every shape has an integral coordinate set,

    $$\{0, 1, 2, \ldots, |S|-1\} = \mathbb{Z}_{|S|} \in \mathcal{Z}(S),$$

    and every rank-$r$ shape has a rank-$r$ coordinate set,

    $$\{(a_0, \ldots, a_{r-1}) \mid a_i \in \mathbb{Z}_{|S_i|}\} = \mathbb{Z}_{(|S_0|,|S_1|,\ldots,|S_{r-1}|)} \in \mathcal{Z}(S).$$

    Note that if shape $P$ coarsens shape $S$, then $\mathcal{Z}(P) \subseteq \mathcal{Z}(S)$. This means that any coordinate within shape $P$ is also a coordinate within shape $S$.
    """)
    return


@app.cell
def _(coord_set):
    # Figure 2: Coordinate sets for three example shapes

    # S = 4
    print("S = 4")
    print("Z(S) = {Z_4}")
    print("Z_4:", coord_set(4))
    print()

    # S = (2, 3)
    print("S = (2, 3)")
    print("Z(S) = {Z_6, Z_(2,3)}")
    print("Z_6:    ", coord_set(6))
    print("Z_(2,3):", coord_set((2, 3)))
    print()

    # S = ((2, 3), 2)
    print("S = ((2, 3), 2)")
    print("Z(S) = {Z_12, Z_(6,2), Z_((2,3),2)}")
    print("Z_12:        ", coord_set(12))
    print("Z_(6,2):     ", coord_set((6, 2)))
    print("Z_((2,3),2): ", coord_set(((2, 3), 2)))
    return


@app.cell
def _(idx2crd):
    # Tabulate the mappings from Figure 2 using idx2crd
    print('S = (2, 3)')
    # S = (2, 3)
    print(f"{'Z_6':>4s}   {'Z_(2,3)':>10s}")
    for _i in range(6):
        _crd = idx2crd(_i, (2, 3))
        print(f'{_i:4d}   {str(_crd):>10s}')
    print()
    print('S = ((2, 3), 2)')
    print(f"{'Z_12':>4s}   {'Z_(6,2)':>10s}   {'Z_((2,3),2)':>16s}")
    for _i in range(12):
    # S = ((2, 3), 2)
        crd_flat = idx2crd(_i, (6, 2))
        crd_nat = idx2crd(_i, ((2, 3), 2))
        print(f'{_i:4d}   {str(crd_flat):>10s}   {str(crd_nat):>16s}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2.2 Coordinates

    In this section, we define classes of coordinates, define a bijection between the compatible coordinate sets of a shape, and provide examples of these coordinate mappings.

    **Definition 2.9.** An *in-bounds coordinate*, or simply *coordinate*, into a shape $S$ is an element of one of its coordinate sets, $c \in \mathbb{Z}_{S'} \in \mathcal{Z}(S)$. Note that a coordinate is always an HTuple($\mathbb{N}$). When intention is clear, we will simply write $c \in \mathcal{Z}(S)$.

    **Definition 2.10.** An *integral coordinate* into a shape $S$ is a coordinate $c \in \mathbb{Z}_{|S|} \in \mathcal{Z}(S)$. Note that an integral coordinate is always an integer, $c \in \mathbb{N}$.

    **Definition 2.11.** A *natural coordinate* into a shape $S$ is a coordinate $\tilde{c} \in \mathbb{Z}_S \in \mathcal{Z}(S)$. Note that a natural coordinate is always an HTuple($\mathbb{N}$) that is congruent to the shape, $\tilde{c} \sim S$.

    To transform between in-bound coordinates, we construct an enumeration over the coordinate sets of a shape $S$ to define coordinate lists. In this work, we choose the colexicographical ordering, $<$, of coordinates defined by:

    $$(a_0, \ldots, a_n) < (b_0, \ldots, b_n) \quad \text{iff} \quad a_n < b_n, \quad \text{or} \quad a_n = b_n \text{ and } (a_0, \ldots, a_{n-1}) < (b_0, \ldots, b_{n-1})$$

    and applied recursively as needed. The colexicographical enumeration defines a bijection on coordinate lists. The function

    $$\text{idx2crd}: \mathbb{Z}_{|S|} \to \mathbb{Z}_{(|S_0|, |S_1|, \ldots, |S_{r-1}|)},$$

    $$i \mapsto \left(i \bmod |S_0|,\ \left\lfloor \frac{i}{|S_0|} \right\rfloor \bmod |S_1|,\ \ldots,\ \left\lfloor \frac{i}{\prod_{k=0}^{r-2} |S_k|} \right\rfloor \right) \qquad (4)$$

    maps the $i$th coordinate of $\mathbb{Z}_{|S|}$ (the $i$th integral coordinate of shape $S$) to the $i$th coordinate of $\mathbb{Z}_{(|S_0|,|S_1|,\ldots,|S_{r-1}|)}$ (the $i$th natural coordinate of shape $(|S_0|, |S_1|, \ldots, |S_{r-1}|)$).

    The inverse of idx2crd is given by

    $$\text{crd2idx}: \mathbb{Z}_{(|S_0|,|S_1|,\ldots,|S_{r-1}|)} \to \mathbb{Z}_{|S|},$$

    $$( c_0, c_1, \ldots, c_{r-1}) \mapsto c_0 + c_1 \cdot |S_0| + \ldots + c_{r-1} \cdot \prod_{k=0}^{r-2} |S_k| \qquad (5)$$

    which maps the $i$th coordinate of $\mathbb{Z}_{(|S_0|,|S_1|,\ldots,|S_{r-1}|)}$ to the $i$th coordinate of $\mathbb{Z}_{|S|}$.
    """)
    return


@app.cell
def _(idx2crd, shape_size):
    # Implement idx2crd and crd2idx from scratch to match the formulas,
    # then verify against the library's idx2crd.
    def my_idx2crd(i, shape):
        """Colexicographic mapping: integer -> natural coordinate tuple."""
        if isinstance(shape, int):
            return i % shape
        coords = []
        for s in shape:
            s_size = shape_size(s)
            coords.append(my_idx2crd(i % s_size, s))
            i = i // s_size
        return tuple(coords)

    def my_crd2idx(coord, shape):
        """Inverse of idx2crd: natural coordinate tuple -> integer."""
        if isinstance(shape, int):
            return coord
        result = 0
        multiplier = 1
        for c, s in zip(coord, shape):
            s_size = shape_size(s)
            result += my_crd2idx(c, s) * multiplier
            multiplier *= s_size
        return result
    print('idx2crd for shape (2, 3):')
    for _i in range(6):
    # Test: shape (2, 3)
        _crd = my_idx2crd(_i, (2, 3))
        _back = my_crd2idx(_crd, (2, 3))
        lib_crd = idx2crd(_i, (2, 3))
        assert _crd == lib_crd, f'Mismatch at i={_i}: {_crd} vs {lib_crd}'
        assert _back == _i
        print(f'  {_i} -> {_crd} -> {_back}')
    print('\nidx2crd for shape ((2, 3), 2):')
    for _i in range(12):
        _crd = my_idx2crd(_i, ((2, 3), 2))
        _back = my_crd2idx(_crd, ((2, 3), 2))
        lib_crd = idx2crd(_i, ((2, 3), 2))
        assert _crd == lib_crd, f'Mismatch at i={_i}: {_crd} vs {lib_crd}'
        assert _back == _i
        print(f'  {_i:2d} -> {str(_crd):>16s} -> {_back:2d}')
    print('\nAll round-trips match!')
    return my_crd2idx, my_idx2crd


@app.cell
def _(Layout, depth, rank, size):
    # Layouts accept any compatible coordinate — demonstrate with concrete layouts
    layout = Layout(((2, 3), 2), ((1, 2), 6))
    # A layout with hierarchical shape ((2, 3), 2)
    print(f'Layout: {layout}')
    print(f'  size = {size(layout)}, rank = {rank(layout)}, depth = {depth(layout)}')
    print()
    print('1D integral coordinates (i):')
    offsets_1d = []
    # Index with 1D integral coordinates
    for _i in range(12):
        o = layout(_i)
        offsets_1d.append(o)
        print(f'  layout({_i:2d}) = {o}')
    print()
    print('2D coarsened coordinates (m, n):')
    offsets_2d = []
    for _n in range(2):
        for _m in range(6):
    # Index with 2D coarsened coordinates (m, n) where m in Z_6, n in Z_2
            o = layout(_m, _n)
            offsets_2d.append(o)
            print(f'  layout({_m}, {_n}) = {o}')
    assert offsets_1d == offsets_2d
    print('\n1D and 2D indexing produce the same offsets!')
    return (layout,)


@app.cell
def _(draw_layout, layout):
    # Visualize the layout to see the coordinate-to-offset mapping
    print("Layout ((2,3),2):((1,2),6) — 2D view")
    draw_layout(layout, colorize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Out-of-bounds Coordinates

    In addition to the coordinate sets we have already defined, it is useful to define coordinates of specific profiles that may not be in the coordinate sets of a shape.

    **Definition 2.12.** An *admissible coordinate* into a shape $S$ is any coordinate $c \in \text{HTuple}(\mathbb{Z})$ that is weakly congruent to the shape, $c \lesssim S$.

    **Definition 2.13.** An *out-of-bounds coordinate* into a shape $S$ is any admissible coordinate $c \in \text{HTuple}(\mathbb{Z})$ that is not in-bounds, $c \notin \mathcal{Z}(S)$.

    **Definition 2.14.** A *congruent coordinate* into a shape $S$ is any coordinate $c \in \text{HTuple}(\mathbb{Z})$ that is congruent to the shape, $c \sim S$. This is denoted as $\overline{\mathbb{Z}}_S = \{c \in \text{HTuple}(\mathbb{Z}) \mid c \sim S\}$.

    Note that `idx2crd` is well-defined for all integers, equivalently coordinates in $\overline{\mathbb{Z}}_{|S|}$, rather than simply the integers in $\mathbb{Z}_{|S|}$. When it is evaluated on an integer $i \ge |S|$, it will always return a coordinate $(c_0, c_1, \ldots, c_{r-1})$ that is out-of-bounds with respect to shape $(|S_0|, |S_1|, \ldots, |S_{r-1}|)$. In contrast, `crd2idx` cannot guarantee an out-of-bounds result for an out-of-bounds coordinate input. Therefore, `crd2idx` and `idx2crd` are only inverses of each other when evaluated on in-bounds coordinates.
    """)
    return


@app.cell
def _(my_crd2idx, my_idx2crd, shape_size):
    # idx2crd is well-defined for out-of-bounds integers
    shape = (2, 3)
    print(f'Shape: {shape}, size = {shape_size(shape)}')
    print()
    print('In-bounds and out-of-bounds idx2crd:')
    for _i in range(10):
        _crd = my_idx2crd(_i, shape)
        _back = my_crd2idx(_crd, shape)
        in_bounds = _i < shape_size(shape)
        round_trips = _back == _i
        print(f"  i={_i}: idx2crd -> {_crd}, crd2idx -> {_back}  {('[in-bounds]' if in_bounds else '[OUT-OF-BOUNDS]')}  {('round-trips' if round_trips else 'NO round-trip')}")
    print()
    oob_coord = (0, 3)
    result = my_crd2idx(oob_coord, shape)
    print(f'crd2idx({oob_coord}, {shape}) = {result}')
    # Show that crd2idx does NOT guarantee out-of-bounds for out-of-bounds input
    print(f'  This maps to {result}, which is IN the range [0, {shape_size(shape)}),')  # out-of-bounds for shape (2, 3): n=3 >= 3
    print(f'  even though {oob_coord} is out-of-bounds for shape {shape}.')
    print(f'  idx2crd({result}, {shape}) = {my_idx2crd(result, shape)} != {oob_coord}')
    print('  So crd2idx and idx2crd are NOT inverses on out-of-bounds coordinates.')
    return


if __name__ == "__main__":
    app.run()
