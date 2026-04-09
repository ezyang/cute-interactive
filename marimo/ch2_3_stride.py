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
    # 2.3 Stride

    The previous section described shapes, their hierarchies, and coordinates for those shapes. To construct layouts of data, threads, or other objects, we define a mapping from coordinates within a shape to offsets.

    **Definition 2.15.** A stride $D$ for a shape $S$ is an HTuple($D$) that is congruent with the shape, $S \sim D$. This stride defines a mapping from a natural coordinate $\tilde{c} \in \mathbb{Z}_S$ to the codomain $D$, given by

    $$\text{inner\_product}: \mathbb{Z} \cdot D \to D, \quad c \cdot d \mapsto cd$$

    $$\text{inner\_product}: \text{HTuple}(\mathbb{Z}) \cdot \text{HTuple}(D) \to D, \quad c \cdot d \mapsto \sum_i \text{inner\_product}(c_i, d_i)$$

    In most cases, strides are also HTuple($\mathbb{Z}$)s, meaning $D = \mathbb{Z}$. The resulting integer produced by inner\_product is typically interpreted as an offset within a data array. However, the concept of a stride element generalizes to any element of an integer-semimodule, which provides significant flexibility in the span of functions that layouts can represent.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, size, cosize, idx2crd, crd2offset

    return Layout, cosize, crd2offset, idx2crd, size


@app.cell
def _(Layout, crd2offset, idx2crd, size):
    # crd2offset computes the inner product of a coordinate with a stride,
    # giving the memory offset. For layout (4, 8) : (1, 4), coordinate (m, n):
    #   offset = m * 1 + n * 4

    shape = (4, 8)
    stride = (1, 4)
    # Example: Layout (4, 8) : (1, 4) — column-major
    L = Layout(shape, stride)
    for _i in range(size(L)):
        _coord = idx2crd(_i, shape)
        _offset = crd2offset(_coord, shape, stride)
        assert _offset == L(_i), f'Mismatch at i={_i}'
    # Show a few examples
    for _i in [0, 1, 5, 22, 31]:
        _coord = idx2crd(_i, shape)
        _offset = crd2offset(_coord, shape, stride)
        print(f'  coord {str(_coord):6s} . stride {stride} = {_offset:3d}  (layout({_i}) = {L(_i)})')
    return


@app.cell
def _(Layout, cosize, crd2offset, idx2crd, size):
    # With hierarchical/nested shapes, crd2offset recurses into sub-tuples.
    # Layout ((2, 2), (4, 2)) : ((1, 8), (2, 16))
    # For a natural coordinate ((a, b), (c, d)):
    #   offset = a*1 + b*8 + c*2 + d*16
    nested_shape = ((2, 2), (4, 2))
    nested_stride = ((1, 8), (2, 16))
    L_nested = Layout(nested_shape, nested_stride)
    print(f'Layout: {L_nested}')
    print(f'  size = {size(L_nested)}, cosize = {cosize(L_nested)}')
    print()
    # Verify crd2offset on nested coordinates
    for _i in range(size(L_nested)):
        _coord = idx2crd(_i, nested_shape)
        _offset = crd2offset(_coord, nested_shape, nested_stride)
        assert _offset == L_nested(_i)
    _i = 22
    c_2d = idx2crd(_i, (4, 8))
    c_nat = idx2crd(_i, nested_shape)
    offset = L_nested(_i)
    # Show the paper's example: integral coordinate 22
    print(f'L(22) = L{c_2d} = L{c_nat} = {offset}')
    print()  # (2, 5) — flat 2D coordinate
    print(f'Breakdown: {c_nat[0][0]}*1 + {c_nat[0][1]}*8 + {c_nat[1][0]}*2 + {c_nat[1][1]}*16')  # ((0, 1), (1, 1)) — natural coordinate
    print(f'         = {c_nat[0][0] * 1} + {c_nat[0][1] * 8} + {c_nat[1][0] * 2} + {c_nat[1][1] * 16} = {offset}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3.1 Integer-Semimodules

    **Definition 2.16.** An integer-semimodule is a set $M$ equipped with an associative addition, $M + M \to M$, and a scalar multiplication, $\mathbb{Z} \cdot M \to M$. For $a, b \in \mathbb{Z}$ and $m, n, p \in M$, the addition and scalar multiplication satisfy

    - Multiplicative Identity: $1 \cdot m = m$,
    - Additive Associativity: $m + (n + p) = (m + n) + p$,
    - Multiplicative Associativity: $a \cdot (b \cdot m) = (ab) \cdot m$.

    Additive identities and inverses are not required, so $(M, +)$ is a semigroup. We write an integer-semimodule as $(M, +, \cdot)$ to denote the set $M$, the additive operation $+$, and the scalar multiplication $\cdot$.

    The integers, $\mathbb{Z}$, are an integer-semimodule.
    The rationals, $\mathbb{Q}$, are an integer-semimodule.
    The field $\mathbb{F}_2 = (\{0, 1\}, \text{XOR}, \text{AND})$ of arithmetic operations modulo 2 is an integer-semimodule. Any Cartesian product or HTuple of integer-semimodules is an integer-semimodule given elementwise addition and scalar multiplication.

    A uniquely useful integer-semimodule is $(\mathbb{Z}_S, +, \cdot)$ with $\mathbb{Z}_S$ the set of all HTuple($\mathbb{Z}$) congruent to $S$. For instance, the basis elements of rank-2 arithmetic tuples form an integer-semimodule:

    $$e_0 = (1, 0), \quad e_1 = (0, 1),$$

    $$\mathbb{Z}_{(\ast,\ast)} = \{a \cdot e_0 + b \cdot e_1 \mid a, b \in \mathbb{Z}\}$$

    with integer scaling and addition defined element-wise:

    $$a \cdot e_0 + b \cdot e_1 = (a, b)$$

    Thus, $e_0$, $e_1$, and any linear combination can be used as strides within a layout. By selecting stride elements from $\mathbb{Z}_S$, layouts can generate natural coordinates of a shape $S$ through the inner\_product operation.
    """)
    return


@app.cell
def _():
    # Demonstrate the integer-semimodule axioms with concrete examples.

    # 1. The integers Z are an integer-semimodule: (Z, +, *)
    m, n, p_val = 5, 3, 7
    a, b = 4, 6

    assert 1 * m == m                                     # Multiplicative identity
    assert m + (n + p_val) == (m + n) + p_val             # Additive associativity
    assert a * (b * m) == (a * b) * m                     # Multiplicative associativity
    print("Z satisfies integer-semimodule axioms. ✓")

    # 2. F2 = ({0, 1}, XOR, AND) is an integer-semimodule
    class F2:
        """Elements of the field F2 with XOR as addition and AND as scalar multiplication."""
        def __init__(self, val):
            self.val = val & 1  # mod 2
        def __add__(self, other):
            return F2(self.val ^ other.val)   # XOR
        def __rmul__(self, scalar):            # Z * F2 -> F2
            return F2((scalar & 1) & self.val) # AND with parity of scalar
        def __eq__(self, other):
            return self.val == other.val
        def __repr__(self):
            return str(self.val)

    x, y, z = F2(1), F2(0), F2(1)
    assert 1 * x == x                         # Multiplicative identity
    assert (x + y) + z == x + (y + z)         # Additive associativity
    assert 3 * (2 * x) == (3 * 2) * x         # Multiplicative associativity
    print("F2 = ({0,1}, XOR, AND) satisfies integer-semimodule axioms. ✓")
    print(f"  Examples: 1 XOR 0 = {(x + y).val}, 1 XOR 1 = {(x + z).val}, 3 * F2(1) = {(3 * x).val}")
    return


@app.cell
def _(idx2crd):
    # The coordinate-tuple semimodule Z_{(*,*)} with basis e0=(1,0), e1=(0,1)
    # Strides can be tuples! A layout with tuple strides generates coordinates.
    def scale(n, elem):
    # This implements the inner_product generalized to tuple-valued strides.
    # When strides are tuples like (1, 0) and (0, 1), the inner_product
    # uses elementwise scaling and addition to produce a coordinate tuple.
        """Scalar multiplication: Z * M -> M"""
        if isinstance(elem, tuple):
            return tuple((scale(n, e) for e in elem))
        return n * elem

    def add(a, b):
        """Addition: M + M -> M"""
        if isinstance(a, tuple) and isinstance(b, tuple):
            return tuple((add(x, y) for x, y in zip(a, b)))
        return a + b

    def inner_product_general(coord, stride):
        """inner_product generalized to semimodule-valued strides."""
        if isinstance(coord, int) and (not isinstance(stride, tuple)):
            return scale(coord, stride)
        if isinstance(coord, int) and isinstance(stride, tuple):
            return scale(coord, stride)
        _result = inner_product_general(coord[0], stride[0])
        for _c, d in zip(coord[1:], stride[1:]):
            _result = add(_result, inner_product_general(_c, d))  # Both are tuples — recurse and sum
        return _result
    e0 = (1, 0)
    e1 = (0, 1)
    shape_coord = (3, 5)
    stride_coord = (e0, e1)
    # Basis vectors of Z_{(*,*)}
    print('Layout with tuple strides: shape (3,5), stride ((1,0), (0,1))')
    print('This generates coordinates as outputs:')
    print()
    # A layout with shape (3, 5) and tuple strides (e0, e1)
    # maps coordinate (i, j) to: i * (1,0) + j * (0,1) = (i, j)
    # This is essentially the identity layout that generates coordinates!
    for _j in range(5):
        for _i in range(3):
            _coord = idx2crd(_i + _j * 3, shape_coord)
            _result = inner_product_general(_coord, stride_coord)
            print(f'  ({_i},{_j}) -> {_result}', end='')
        print()
    return (inner_product_general,)


@app.cell
def _(inner_product_general):
    # A more interesting example: strides can be non-basis tuples.
    # With stride ((2, 0), (0, 3)), coordinate (i, j) maps to (2i, 3j).
    # This generates scaled coordinates — useful for strided access patterns.
    stride_scaled = ((2, 0), (0, 3))
    print('Layout with scaled tuple strides: shape (3,5), stride ((2,0), (0,3))')
    print('Maps (i,j) -> (2i, 3j):')
    print()
    for _j in range(5):
        row = []
        for _i in range(3):
            _coord = (_i, _j)
            _result = inner_product_general(_coord, stride_scaled)
            row.append(_result)
        print(f'  j={_j}: {row}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The key insight of integer-semimodules is that the same inner\_product mechanism used for ordinary integer strides generalizes seamlessly to other algebraic structures. By choosing strides from $\mathbb{Z}_{(\ast,\ast)}$, layouts can produce coordinates as outputs rather than integer offsets. By choosing strides from $\mathbb{F}_2$, layouts can produce XOR-based swizzle patterns.

    This uniformity — the same shape/stride/inner\_product framework regardless of the codomain — is what makes CuTe layouts a powerful abstraction that goes well beyond traditional row-major/column-major indexing.
    """)
    return


if __name__ == "__main__":
    app.run()
