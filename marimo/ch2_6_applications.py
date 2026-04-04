import marimo

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.22.0",
#     "matplotlib",
#     "numpy",
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
    # 2.6 Applications

    CUTE provides a compact representation for a set of layouts that is strictly larger than can be represented with traditional flat shapes and strides. In contrast, libraries like CUTLASS v2 implement each layout individually and manually. This approach is labor-intensive, error-prone, and requires significant development time. To illustrate the complexity, the CUTLASS v2 code base contains nearly 300 separately implemented layouts spread across 87 files, collectively amounting to approximately 55,000 lines of code. Furthermore, many algorithms in CUTLASS v2 are designed to operate with only a limited subset of these layouts, exacerbating maintenance and scalability challenges. By comparison, CUTE's core layout representation, along with the associated algebra for manipulating layouts, requires only 3,000 lines of code and is capable of representing all 300 layouts found in CUTLASS v2 and more. Algorithms implemented in CUTE can enforce constraints on the rank or shape of their input, but remain compatible with any layout that satisfies these preconditions. This decoupling of algorithm logic from specific data or thread layouts results in more flexible and composable code.

    The benefits above were recognized and CUTE now forms the basis of CUTLASS v3, CUTLASS v4, and CuTe DSL, which are all built on top of CUTE's core layout representation and algebra. The complex data layouts and partitioning patterns required by modern tensor instructions are represented and manipulated with CUTE's single, consistent, and composable representation that has been robust across multiple generations of NVIDIA's instruction set architecture.

    In this section, we detail the use of only the layout and tensor representation to provide powerful generic implementations of two of the most fundamental algorithms: COPY and GEMM. These algorithms are implemented with CUTE tensors and are used to illustrate logical implementations being applicable to a wide range of applications. These algorithms, though often optimal on their own, serve as excellent generic reference implementations for optimized versions. The layout algebra in Section 3 provides methods for inspecting and manipulating layouts to perform these optimizations.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, size, mode, cosize, rank
    from tensor_layouts.viz import draw_layout

    return Layout, cosize, draw_layout, mode, size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.6.1 COPY

    A generic COPY algorithm written with CUTE tensors is

    ```python
    # @pre size(src) == size(dst)
    def copy(src : Tensor,       # N
             dst : Tensor):       # N
        for i in range(size(dst)):
            dst[i] = src[i]
    ```

    where the precondition specifies that the size of the tensors is equal. Equivalently written in the tensor argument comments, both tensors are compatible with a shape N.

    This simple implementation of COPY accommodates a wide range of applications by varying the layouts of the source and destination tensors. Table 2 provides examples of common applications and their associated source and destination layouts.

    | Application | Source Layout | Destination Layout |
    |---|---|---|
    | 1D Arrays | $8 : 1$ | $8 : 1$ |
    | ND Arrays | $(8, 2, 3) : (1, 16, 32)$ | $(8, 2, 3) : (1, 16, 32)$ |
    | Gather | $(2, 3, 2) : (42, 1, 128)$ | $12 : 1$ |
    | Scatter | $12 : 1$ | $(2, 3, 2) : (42, 1, 128)$ |
    | Broadcast | $7 : 0$ | $7 : 1$ |
    | Constant | $7 : 0$ | $7 : 0$ |
    | Transpose | $(8, 3) : (1, 8)$ | $(8, 3) : (3, 1)$ |
    | Tensor Transpose | $(8, (3, 5)) : (1, (57, 8))$ | $(8, 15) : (1, 8)$ |
    """)
    return


@app.cell
def _(size):
    # Implement the generic COPY using layouts to compute offsets into data arrays.
    # Since tensor_layouts.Tensor doesn't hold mutable data, we use plain lists
    # with layouts providing the index mapping.
    def copy(src_data, src_layout, dst_data, dst_layout):
        """Generic COPY: dst[dst_layout(i)] = src[src_layout(i)] for all i."""
        assert size(src_layout) == size(dst_layout)
        for _i in range(size(dst_layout)):
            dst_data[dst_layout(_i)] = src_data[src_layout(_i)]

    return (copy,)


@app.cell
def _(Layout, copy):
    # 1D Arrays: 8:1 -> 8:1  (simple contiguous copy)
    src_layout = Layout(8, 1)
    dst_layout = Layout(8, 1)
    _src_data = list(range(10, 18))  # [10, 11, ..., 17]
    _dst_data = [0] * 8
    copy(_src_data, src_layout, _dst_data, dst_layout)
    assert _dst_data == _src_data
    print(f'1D Arrays:  src={_src_data}')
    print(f'            dst={_dst_data}')
    return


@app.cell
def _(Layout, copy, cosize):
    src_layout_1 = Layout((2, 3, 2), (42, 1, 128))
    dst_layout_1 = Layout(12, 1)
    src_size = cosize(src_layout_1)
    _src_data = [f'x{_i}' for _i in range(src_size)]
    _dst_data = [None] * 12
    copy(_src_data, src_layout_1, _dst_data, dst_layout_1)
    print(f'Gather: src offsets = {[src_layout_1(_i) for _i in range(12)]}')
    print(f'        dst = {_dst_data}')
    for _i in range(12):
        assert _dst_data[_i] == _src_data[src_layout_1(_i)]
    return


@app.cell
def _(Layout, copy, cosize):
    src_layout_2 = Layout(12, 1)
    dst_layout_2 = Layout((2, 3, 2), (42, 1, 128))
    _src_data = [f'v{_i}' for _i in range(12)]
    dst_size = cosize(dst_layout_2)
    _dst_data = [None] * dst_size
    copy(_src_data, src_layout_2, _dst_data, dst_layout_2)
    print(f'Scatter: dst offsets = {[dst_layout_2(_i) for _i in range(12)]}')
    print(f'         values at those offsets: {[_dst_data[dst_layout_2(_i)] for _i in range(12)]}')
    for _i in range(12):
        assert _dst_data[dst_layout_2(_i)] == _src_data[_i]
    return


@app.cell
def _(Layout, copy):
    src_layout_3 = Layout(7, 0)
    dst_layout_3 = Layout(7, 1)
    _src_data = [42]
    _dst_data = [0] * 7
    copy(_src_data, src_layout_3, _dst_data, dst_layout_3)
    print(f'Broadcast: src has 1 element = {_src_data[0]}')
    print(f'           dst = {_dst_data}')
    assert _dst_data == [42] * 7
    return


@app.cell
def _(Layout, copy):
    src_layout_4 = Layout((8, 3), (1, 8))
    dst_layout_4 = Layout((8, 3), (3, 1))
    _src_data = list(range(24))
    _dst_data = [0] * 24
    copy(_src_data, src_layout_4, _dst_data, dst_layout_4)
    for _i in range(8):
        for _j in range(3):
            assert _dst_data[dst_layout_4(_i, _j)] == _src_data[src_layout_4(_i, _j)]
    print('Transpose (column-major -> row-major):')
    print(f'  src layout: {src_layout_4}')
    print(f'  dst layout: {dst_layout_4}')
    print(f'  src[2,1] at offset {src_layout_4(2, 1)} = {_src_data[src_layout_4(2, 1)]}')
    print(f'  dst[2,1] at offset {dst_layout_4(2, 1)} = {_dst_data[dst_layout_4(2, 1)]}')
    assert _src_data[src_layout_4(2, 1)] == _dst_data[dst_layout_4(2, 1)]
    return dst_layout_4, src_layout_4


@app.cell
def _(draw_layout, dst_layout_4, src_layout_4):
    # Visualize the transpose: source (col-major) vs destination (row-major)
    print('Source layout (column-major): (8,3):(1,8)')
    draw_layout(src_layout_4, colorize=True)
    print('\nDestination layout (row-major): (8,3):(3,1)')
    draw_layout(dst_layout_4, colorize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Any tensor of any rank can be copied to any tensor of any other rank. In that sense, COPY is a rank-1 algorithm regardless of the arguments' ranks. This is a version of rank-agnostic programming.

    When the `idx2crd` function for the source and destination tensors is computationally inexpensive to evaluate (e.g., when the tensor shapes are statically known), the above implementation never actually generates dynamic coordinate transformations. In such cases, the loop can be unrolled, `idx2crd` can be statically applied to the loop index `i`, and the `inner_product` computation incurs minimal overhead in computing offsets. This is a version of static analysis and optimization since the layout shape and/or strides are often known at compile-time and available to the compiler. If `idx2crd` does incur a runtime cost, the provided implementation still serves as a robust reference to validate optimized versions that may further inspect domains and transform layouts using operations detailed in Section 3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.6.2 GEMM

    A generic GEMM algorithm written with CUTE tensors is

    ```python
    # @pre M: size[0](A) == size[0](C)
    # @pre N: size[0](B) == size[1](C)
    # @pre K: size[1](A) == size[1](B)
    def gemm(A : Tensor,       # (M,K)
             B : Tensor,       # (N,K)
             C : Tensor):      # (M,N)
        for k in range(size[1](B)):
            for n in range(size[0](B)):
                for m in range(size[0](A)):
                    C[m,n] += A[m,k] * B[n,k]
    ```

    where the precondition specifies the logical constraints of the GEMM algorithm. In the comments of each tensor parameter, we write the shape that each tensor must be compatible with.

    This simple implementation of GEMM (and a batched-GEMM extension) can encompass a variety of applications by varying the layouts of the tensors. In Table 3, some common applications and example layouts are shown. These include all of the N-T variants of the BLAS GEMM and the generically strided (dm*, dn*, dk*) variants of the BLIS GEMM. This also functions as a fully generic tensor-tensor contraction (GETT) where the tensors are folded into the appropriate matrix shape by grouping logical row modes, column modes, reduction modes, and batched modes. Creating a layout (as a functional composition of CUTE layouts) that implements the im2col transformation also allows GEMM to implement CONV, which is core to many modern machine learning applications.

    | Application | A-Layout | B-Layout | C-Layout |
    |---|---|---|---|
    | NT GEMM | $(M, K) : (1, \text{lda})$ | $(N, K) : (1, \text{ldb})$ | $(M, N) : (1, \text{ldc})$ |
    | TN GEMM | $(M, K) : (\text{lda}, 1)$ | $(N, K) : (\text{ldb}, 1)$ | $(M, N) : (1, \text{ldc})$ |
    | NTT GEMM | $(N, K) : (1, \text{ldb})$ | $(M, K) : (1, \text{lda})$ | $(N, M) : (1, \text{ldc})$ |
    | BLIS GEMM | $(M, K) : (\text{dma}, \text{dka})$ | $(N, K) : (\text{dnb}, \text{dkb})$ | $(M, N) : (\text{dmc}, \text{dnc})$ |
    | GETT | $((M_1, M_2), K) : ((1, W), X)$ | $(N, K) : (K, 1)$ | $((M_1, M_2), N) : ((1, Y), Z)$ |
    | GETT | $(M, (K_1, K_2)) : ((W, X), 1)$ | $(N, (K_1, K_2)) : ((Y, Z), 1)$ | $(M, N) : (1, M)$ |
    | CONV | $(K, (C, T, R, S)) : D_A$ | $((N, Z, P, Q), (C, T, R, S)) : D_B$ | $(K, (N, Z, P, Q)) : D_C$ |

    By abstracting the fused-multiply-add operation and providing sufficiently powerful tiling utilities, this algorithm can be adapted and applied recursively at each level of an architectural hierarchy.
    """)
    return


@app.cell
def _(mode, size):
    # Implement the generic GEMM using layouts to compute offsets into data arrays.
    def gemm(A_data, A_layout, B_data, B_layout, C_data, C_layout):
        """Generic GEMM: C += A * B^T via layouts.
        A is (M, K), B is (N, K), C is (M, N)."""
        M = size(mode(A_layout, 0))
        N = size(mode(B_layout, 0))
        K = size(mode(A_layout, 1))
        assert M == size(mode(C_layout, 0)), f'M mismatch: {M} vs {size(mode(C_layout, 0))}'
        assert N == size(mode(C_layout, 1)), f'N mismatch: {N} vs {size(mode(C_layout, 1))}'  # Precondition checks
        assert K == size(mode(B_layout, 1)), f'K mismatch: {K} vs {size(mode(B_layout, 1))}'
        for k in range(K):
            for n in range(N):
                for m in range(M):
                    C_data[C_layout(m, n)] = C_data[C_layout(m, n)] + A_data[A_layout(m, k)] * B_data[B_layout(n, k)]

    return (gemm,)


@app.cell
def _(Layout, gemm):
    import numpy as np
    M, N, K = (4, 3, 5)
    # NT GEMM: A is column-major (M,K):(1,lda), B is column-major (N,K):(1,ldb)
    # C is column-major (M,N):(1,ldc)
    # This corresponds to C = A @ B^T in standard notation.
    _lda, _ldb, _ldc = (M, N, M)
    A_layout = Layout((M, K), (1, _lda))  # leading dimensions = number of rows (col-major)
    B_layout = Layout((N, K), (1, _ldb))
    C_layout = Layout((M, N), (1, _ldc))  # column-major
    print(f'NT GEMM: M={M}, N={N}, K={K}')  # column-major
    print(f'  A: {A_layout}')  # column-major
    print(f'  B: {B_layout}')
    print(f'  C: {C_layout}')
    np.random.seed(42)
    A_np = np.random.randn(M, K)
    B_np = np.random.randn(N, K)
    A_data = list(A_np.T.flatten())
    # Create data in column-major order
    B_data = list(B_np.T.flatten())
    C_data = [0.0] * (M * N)
    for _i in range(M):
        for _j in range(K):
    # Flatten to column-major storage (Fortran order)
            assert A_data[A_layout(_i, _j)] == A_np[_i, _j]  # column-major
    gemm(A_data, A_layout, B_data, B_layout, C_data, C_layout)
    C_ref = A_np @ B_np.T
    C_result = np.array([[C_data[C_layout(_i, _j)] for _j in range(N)] for _i in range(M)])
    # Verify layout gives correct access
    assert np.allclose(C_result, C_ref)
    # Reference: C = A @ B^T
    # Extract result from layout
    print('\nNT GEMM result matches numpy reference!')
    return A_layout, A_np, B_layout, B_np, C_layout, C_ref, K, M, N, np


@app.cell
def _(A_np, B_np, C_ref, K, Layout, M, N, gemm, np):
    # TN GEMM: A is row-major (M,K):(lda,1), B is row-major (N,K):(ldb,1)
    # C is column-major (M,N):(1,ldc)
    _lda, _ldb, _ldc = (K, K, M)
    A_layout_tn = Layout((M, K), (_lda, 1))
    B_layout_tn = Layout((N, K), (_ldb, 1))  # row-major
    C_layout_tn = Layout((M, N), (1, _ldc))  # row-major
    print(f'TN GEMM: M={M}, N={N}, K={K}')  # column-major
    print(f'  A: {A_layout_tn}')
    print(f'  B: {B_layout_tn}')
    print(f'  C: {C_layout_tn}')
    A_data_tn = list(A_np.flatten())
    B_data_tn = list(B_np.flatten())
    C_data_tn = [0.0] * (M * N)
    # Flatten to row-major storage (C order)
    gemm(A_data_tn, A_layout_tn, B_data_tn, B_layout_tn, C_data_tn, C_layout_tn)  # row-major
    C_result_tn = np.array([[C_data_tn[C_layout_tn(_i, _j)] for _j in range(N)] for _i in range(M)])
    assert np.allclose(C_result_tn, C_ref)
    print('\nTN GEMM result matches numpy reference!')
    return


@app.cell
def _(Layout, gemm, np):
    # GETT: tensor contraction via GEMM with hierarchical layouts.
    # A has shape ((M1, M2), K) — the row mode is a multi-mode.
    # This shows that GEMM naturally handles tensor contractions
    # when tensors are folded into matrix form with hierarchical shapes.

    M1, M2, K_gett, N_gett = 3, 2, 4, 5
    M_total = M1 * M2  # = 6

    # A: ((M1,M2), K) with hierarchical row mode
    # Column-major-ish: stride ((1, M1), M1*M2)
    A_layout_gett = Layout(((M1, M2), K_gett), ((1, M1), M1 * M2))
    # B: (N, K) standard column-major
    B_layout_gett = Layout((N_gett, K_gett), (1, N_gett))
    # C: ((M1,M2), N) with hierarchical row mode
    C_layout_gett = Layout(((M1, M2), N_gett), ((1, M1), M1 * M2))

    print(f"GETT: M=({M1},{M2})={M_total}, N={N_gett}, K={K_gett}")
    print(f"  A: {A_layout_gett}")
    print(f"  B: {B_layout_gett}")
    print(f"  C: {C_layout_gett}")

    np.random.seed(7)
    A_flat = np.random.randn(M_total * K_gett)
    B_flat = np.random.randn(N_gett * K_gett)
    C_data_gett = [0.0] * (M_total * N_gett)

    gemm(list(A_flat), A_layout_gett, list(B_flat), B_layout_gett,
         C_data_gett, C_layout_gett)

    # Reference: reshape flat data according to layouts, do standard matmul
    A_mat = np.array([[A_flat[A_layout_gett(m, k)] for k in range(K_gett)]
                      for m in range(M_total)])
    B_mat = np.array([[B_flat[B_layout_gett(n, k)] for k in range(K_gett)]
                      for n in range(N_gett)])
    C_ref_gett = A_mat @ B_mat.T

    C_result_gett = np.array([[C_data_gett[C_layout_gett(m, n)] for n in range(N_gett)]
                              for m in range(M_total)])
    assert np.allclose(C_result_gett, C_ref_gett)
    print("\nGETT result matches reference — hierarchical layouts work seamlessly!")
    return (A_layout_gett,)


@app.cell
def _(A_layout, B_layout, C_layout, draw_layout):
    # Visualize the GEMM layouts for the NT case
    print("NT GEMM layouts (M=4, N=3, K=5):")
    print("\nA (M,K) = (4,5):(1,4) — column-major")
    draw_layout(A_layout, colorize=True)

    print("\nB (N,K) = (3,5):(1,3) — column-major")
    draw_layout(B_layout, colorize=True)

    print("\nC (M,N) = (4,3):(1,4) — column-major")
    draw_layout(C_layout, colorize=True)
    return


@app.cell
def _(A_layout_gett, draw_layout):
    # Visualize the GETT layout — note the hierarchical row mode
    print("GETT A-layout: ((3,2),4):((1,3),6) — hierarchical M mode")
    draw_layout(A_layout_gett, colorize=True)
    return


if __name__ == "__main__":
    app.run()
