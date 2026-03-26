# CuTe Layout Interactive Notebooks

## Project Goal
Generate executable Jupyter notebooks (one per section) illustrating examples from the CuTe Layout paper (March 2026 preprint, `cute-layout.pdf`) using the `tensor-layouts` Python library.

## Notebook Conventions
- **Verbatim paper text**: Markdown cells should contain the paper's text as-is (not rephrased). Drop into code cells only when they add interactive value.
- **Symbolic `e`**: Use an opaque `E` class for pure expressions `e(m)`, `e(m,n)` etc. The class supports equality by arguments so assertions work. When the paper defines a derived expression like `g(i) = e(3*i+2)`, unfold it explicitly as `g = lambda i: e(3*i + 2)`.
- **Executable loops**: When the paper shows C loop examples, make them into executable Python loops (writing to dicts) alongside the Layout-based equivalent, asserting equality.
- **Don't jump ahead**: Only include code for concepts introduced in the current section. If the paper poses a question that will be answered later, leave it as prose.
- **Concrete parameters**: When the paper uses symbolic constants (like `p, q`), pick concrete non-trivial values (avoid 1) so strides are visibly distinct.
- **Naming**: `ch{X}_{Y}_{short_name}.ipynb` e.g. `ch1_2_canonical_loops.ipynb`
- **Visualization**: Use only `tensor-layouts` viz (`draw_layout`/`show_layout`) for all diagrams — never matplotlib directly. Use `colorize=True` and `color_layout` to customize coloring (e.g., grouping pairs of physical elements). When the paper shows physical data, visualize it as a 1D layout alongside the logical views.

## Environment
- Python venv at `.venv/` (managed by `uv`; invoke as `uv pip install --python .venv/bin/python <pkg>`)
- `tensor-layouts` v0.1.1 installed (pure-Python CuTe layout algebra, MIT, by Meta)
- `pymupdf` installed for PDF reading
- `jupyter`, `matplotlib` installed
- PDF: `cute-layout.pdf`

## Reading the PDF
```python
import fitz
doc = fitz.open('cute-layout.pdf')
print(doc[3].get_text())  # 0-indexed page number
```

## Testing Notebooks
```bash
.venv/bin/jupyter nbconvert --to notebook --execute ch1_2_canonical_loops.ipynb --output /tmp/test.ipynb
```

## Completed Sections
- [x] 1.2 Canonical Loops and Loop Transformations (`ch1_2_canonical_loops.ipynb`)
- [x] 1.3 Tensors and Folding (`ch1_3_tensors_and_folding.ipynb`)
- [x] 2.1 Tuples and HTuples (`ch2_1_tuples_and_htuples.ipynb`)
- [x] 2.2 Shape (`ch2_2_shape.ipynb`)
- [x] 2.3 Stride (`ch2_3_stride.ipynb`)
- [x] 2.4 Layout (`ch2_4_layout.ipynb`)
- [x] 2.5 Tensor (`ch2_5_tensor.ipynb`)
- [x] 2.6 Applications (`ch2_6_applications.ipynb`)
- [x] 3.1 Concatenate (`ch3_1_concatenate.ipynb`)
- [x] 3.2 Coalesce (`ch3_2_coalesce.ipynb`)
- [x] 3.3 Composition (`ch3_3_composition.ipynb`)
- [x] 3.4 Inverse (`ch3_4_inverse.ipynb`)
- [x] 3.5 Complement (`ch3_5_complement.ipynb`)

## tensor-layouts Library Reference

### Installation & Imports
```python
from tensor_layouts import Layout, Tile, Swizzle, make_swizzle, Tensor
from tensor_layouts import compose, complement, logical_divide, logical_product
from tensor_layouts import size, cosize, rank, depth, mode, flatten, coalesce
from tensor_layouts import idx2crd, crd2flat, crd2offset
from tensor_layouts import zipped_divide, tiled_divide, flat_divide
from tensor_layouts import zipped_product, tiled_product, blocked_product, raked_product, flat_product
from tensor_layouts import right_inverse, left_inverse, nullspace, max_common_layout
from tensor_layouts import shape_div, shape_mod, prefix_product, suffix_product
from tensor_layouts.layout_utils import make_ordered_layout, tile_to_shape
from tensor_layouts.viz import draw_layout, show_layout  # show_* returns Figure
```

### Layout Construction
```python
Layout((4, 8), (1, 4))         # 4x8 column-major
Layout((4, 8), (8, 1))         # 4x8 row-major
Layout((4, 8))                 # shape-only => column-major strides auto
Layout(32, 1)                  # 1D, 32 contiguous elements
Layout(((2,4), 8), ((1,2), 8)) # hierarchical/nested shape
Layout(layout_a, layout_b)     # bundle two layouts as modes
```

### Calling / Indexing
```python
layout = Layout((4, 8), (1, 4))
layout(2, 3)       # -> 14 (offset = 2*1 + 3*4)
layout(None, 3)    # -> Layout(4,):(1,) — slice: fix dim 1, keep dim 0
```

### Key Query Functions
- `size(L)` — total elements (product of shape)
- `cosize(L)` — memory span (max offset + 1)
- `rank(L)` — number of top-level modes
- `depth(L)` — nesting depth
- `mode(L, i)` — extract mode i as sub-layout

### Core Algebra
- `compose(A, B)` — function composition: `compose(A,B)(i) = A(B(i))`
- `complement(L, bound)` — fills stride gaps
- `logical_divide(L, tiler)` — factor into (tile, rest)
- `logical_product(A, B)` — replicate A across B positions

### Division Variants
| Function | Result structure |
|----------|-----------------|
| `logical_divide(L, T)` | `((TileM, RestM), (TileN, RestN))` |
| `zipped_divide(L, T)` | `((TileM, TileN), (RestM, RestN))` |
| `tiled_divide(L, T)` | `((TileM, TileN), RestM, RestN)` |
| `flat_divide(L, T)` | `(TileM, TileN, RestM, RestN)` |

### Product Variants
| Function | Result structure |
|----------|-----------------|
| `logical_product(A, B)` | `(A, composed_complement)` |
| `blocked_product(A, B)` | `((A0,B0), (A1,B1))` — A varies fastest |
| `raked_product(A, B)` | `((B0,A0), (B1,A1))` — B varies fastest |

### Layout Manipulation
- `flatten(L)` — remove hierarchy, flat rank-N
- `coalesce(L)` — merge contiguous modes
- `sort(L)` — sort modes by stride
- `append/prepend(L, other)` — add mode
- `upcast(L, n)` / `downcast(L, n)` — change element granularity

### Visualization
```python
from tensor_layouts.viz import draw_layout
draw_layout(layout)                          # inline in Jupyter
draw_layout(layout, colorize=True)           # rainbow colors
draw_layout(layout, color_by="row")          # color by row/column/offset
draw_layout(layout, flatten_hierarchical=False)  # show nested coords in cells
draw_layout(layout, filename="out.svg")      # save to file
```

### CuTe C++ ↔ Python Mapping
| CuTe C++ | Python |
|----------|--------|
| `Layout<Shape, Stride>` | `Layout(shape, stride)` |
| `make_layout(A, B)` | `Layout(A, B)` |
| `composition(A, B)` | `compose(A, B)` |
| `get<I>(L)` | `mode(L, I)` |
| `Tensor{ptr, layout}` | `Tensor(layout, offset=0)` |

### Tile and Swizzle
```python
Tile(Layout(3, 4), Layout(8, 2))   # mode-by-mode composition helper
Swizzle(bits=3, base=0, shift=3)   # XOR-based bank conflict avoidance
compose(Swizzle(3,0,3), Layout((8,8),(8,1)))  # swizzled layout
```
