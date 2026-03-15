---
name: helion-docs
description: Guide for writing GPU kernels in Helion, a PyTorch-embedded DSL that compiles to Triton. Auto-activate when the user asks to write, implement, fix, or understand a Helion kernel, or when working with helion.language constructs (hl.tile, hl.zeros, hl.dot, etc.), or when they mention "helion kernel" or want to convert PyTorch code to a Helion kernel. Also use when the user is debugging kernel compilation errors or needs help with Helion's tiling model.
---

# Writing Helion Kernels

You are writing GPU kernels in Helion, a Python-embedded DSL that compiles to Triton. Helion kernels look like PyTorch code with explicit tiling. This skill covers how to structure kernels, use the language API, and avoid common pitfalls.

## Default: Skip Autotuning During Development

When writing or iterating on kernels, always default to `autotune_effort="none"`. Autotuning is expensive (5-15 minutes) and unnecessary for verifying correctness. Only autotune when the user explicitly asks for performance tuning.

```python
@helion.kernel(autotune_effort="none")
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    ...
```

Or provide an explicit config to skip autotuning:
```python
@helion.kernel(config=helion.Config(block_sizes=[64, 64]))
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    ...
```

## Imports and Conventions

Always use this import pattern:
```python
from __future__ import annotations
import torch
import helion
import helion.language as hl
```
Never `import helion as hl`. The module `helion.language` is aliased to `hl`.

## Kernel Structure

A Helion kernel has three sections:

1. **Host code** (before any `hl.tile` loop): standard PyTorch on CPU. Allocate outputs, compute shapes, reshape inputs.
2. **Grid loop** (outermost `hl.tile`): maps to GPU thread blocks executing in parallel.
3. **Device code** (inside grid loop): compiled to a single Triton kernel. Use PyTorch ops on tiles.

```python
@helion.kernel(autotune_effort="none")
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()                          # Host code
    out = torch.empty_like(x)                # Host code
    for tile_m, tile_n in hl.tile([m, n]):   # Grid loop
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2  # Device code
    return out
```

One Helion kernel always compiles to exactly one GPU kernel. The code outside `hl.tile` loops runs on CPU; code inside runs on GPU.

## Tiling with `hl.tile`

`hl.tile(sizes)` subdivides an iteration space into tiles executed in parallel. Tile sizes are determined by autotuning or explicit `Config`.

```python
# 1D tiling
for tile in hl.tile(n):
    out[tile] = x[tile] + 1

# 2D tiling
for tile_m, tile_n in hl.tile([m, n]):
    out[tile_m, tile_n] = x[tile_m, tile_n]

# Explicit block size
for tile in hl.tile(n, block_size=64):
    out[tile] = x[tile]

# Nested: outer = grid (parallel), inner = sequential loop within each block
for tile_m in hl.tile(m):
    acc = hl.zeros([tile_m], dtype=torch.float32)
    for tile_n in hl.tile(n):              # Sequential reduction loop
        acc += torch.sum(x[tile_m, tile_n], dim=1)
    out[tile_m] = acc
```

## Indexing Rules (Critical Difference from PyTorch)

**Helion requires explicit indices for every dimension of a tensor.** Unlike PyTorch where `k[i]` on a 3D tensor returns a 2D slice, Helion raises `RankMismatch` if the number of indices doesn't match the tensor's rank.

```python
# x is shape [batch, height, width], k is shape [batch, kh, kw]

# WRONG — too few indices for 3D tensor k:
z[i, h_i, w_i] = (patch * k[i]).sum(1).sum(1)
# Raises: helion.exc.RankMismatch: Expected ndim=3, but got ndim=1. You have too few indices.

# CORRECT — provide all 3 indices explicitly:
z[i, h_i, w_i] = (patch * k[i, :, :]).sum(1).sum(1)
```

This applies to all tensor accesses inside device code. When you want "all elements along a dimension," you must write the explicit `:` slice. The rule is simple: **count the tensor's dimensions and provide that many indices**.

Tile indexing preserves dimensions: `x[tile_m, tile_n]` loads a 2D block, not a scalar. A tile variable like `tile_m` occupies one index dimension. Scalar indices from `range()` or `hl.grid()` also occupy one dimension. Slices (`:`, `h_i:h_i+kh`) occupy one dimension. The total must equal the tensor's `ndim`.

```python
# 2D tensor, need exactly 2 indices:
x[tile_m, tile_n]       # OK: tile, tile
x[tile_m, :]            # OK: tile, slice
x[tile_m, 0]            # OK: tile, scalar

# 3D tensor, need exactly 3 indices:
x[tile_b, tile_m, :]    # OK: tile, tile, slice
x[tile_b, tile_m, tile_n]  # OK: tile, tile, tile
```

## `hl.grid` vs `hl.tile`

`hl.grid(n)` iterates over scalar indices (equivalent to `hl.tile(n, block_size=1)`). Use it when you need per-element control rather than vectorized tile ops.

```python
for i in hl.grid(n):
    out[i] = x[i] + 1  # Scalar index

for i, j in hl.grid([m, n]):
    out[i, j] = compute(i, j)
```

Prefer `hl.tile` for performance. Use `hl.grid` when the algorithm requires scalar indexing (e.g., iterating over batch indices with complex per-element logic).

## Tile Attributes

Tile objects expose useful properties:
- `tile.begin` / `tile.end`: start/end of the tile range
- `tile.block_size`: size of the tile
- `tile.id`: ordinal tile ID (0, 1, 2, ...)
- `tile.index`: the index range `[begin, begin+1, ..., end-1]`

```python
for tile in hl.tile(n, block_size=block_size):
    partial[tile.id] = x[tile].sum()       # Store per-tile result
    # Nested tile from tile range
    for inner in hl.tile(tile.begin, tile.end):
        ...
```

## Tensor Creation on Device

Inside `hl.tile` loops, create tile-sized tensors with:

```python
acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)    # Zero-initialized accumulator
max_val = hl.full([tile_m], float("-inf"), dtype=torch.float32)  # Constant fill
```

These create register-resident tensors sized to the tile dimensions. Always use `torch.float32` for accumulators to avoid precision loss.

## Reductions

Reductions along a dimension use standard PyTorch ops inside tile loops:

```python
# Sum reduction over columns
for tile_m in hl.tile(m):
    acc = hl.zeros([tile_m], dtype=torch.float32)
    for tile_n in hl.tile(n):
        acc += torch.sum(x[tile_m, tile_n], dim=1)  # Reduce dim=1 within tile
    out[tile_m] = acc
```

The pattern is: accumulator outside inner loop, reduce within inner loop, store after.

**Important: chain single-dimension reductions.** Helion requires reducing one dimension at a time. Use `.sum(1).sum(1)` not `.sum([1, 2])`.

For multi-pass algorithms (like softmax), use multiple sequential inner loops:

```python
for tile_m in hl.tile(m):
    # Pass 1: find max
    max_val = hl.full([tile_m], float("-inf"), dtype=torch.float32)
    for tile_n in hl.tile(n):
        max_val = torch.maximum(max_val, torch.amax(x[tile_m, tile_n], dim=1))
    # Pass 2: compute exp sum
    denom = hl.zeros([tile_m], dtype=torch.float32)
    for tile_n in hl.tile(n):
        denom += torch.exp(x[tile_m, tile_n] - max_val[:, None]).sum(dim=1)
    # Pass 3: normalize
    for tile_n in hl.tile(n):
        out[tile_m, tile_n] = torch.exp(x[tile_m, tile_n] - max_val[:, None]) / denom[:, None]
```

## Matrix Multiplication

For matmul, use `torch.addmm` (fuses add+mm) or `hl.dot` (lower-level, more control):

```python
# Standard matmul pattern with torch.addmm
for tile_m, tile_n in hl.tile([m, n]):
    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    for tile_k in hl.tile(k):
        acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
    out[tile_m, tile_n] = acc
```

`hl.dot` gives more control and supports FP8/int8 inputs, explicit `out_dtype`, and `acc` parameter:
```python
acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
acc = hl.dot(x_fp8, y_fp8, acc=acc, out_dtype=torch.float32)
```

Use `hl.dot` when you need FP8 support, explicit output dtype control, or architecture-specific matmul behavior. Use `torch.addmm`/`torch.baddbmm` for standard float16/bfloat16/float32 matmuls.

## Explicit Load/Store and Masking

Usually, indexing tensors with tiles (`x[tile]`) handles loads/stores automatically. For fine-grained control:

```python
val = hl.load(x, [tile], eviction_policy="evict_last")
hl.store(out, [tile_m, tile_n], result)
```

Use `extra_mask` for conditional loads (boundary checking, ragged access):
```python
# Load with bounds checking
idx = base + hl.arange(BLOCK)
mask = idx < n
values = hl.load(x, [idx], extra_mask=mask)  # Out-of-bounds elements are zeroed
hl.store(out, [idx], values, extra_mask=mask)
```

## Specialize and Constexpr

`hl.specialize(val)` makes a runtime value a compile-time constant, triggering recompilation when it changes. Use for performance-critical dimensions like `head_dim` in attention:

```python
head_dim = hl.specialize(q.size(-1))  # Baked into generated code
```

`hl.constexpr` marks a kernel parameter as compile-time constant. Each distinct value compiles a separate kernel variant:

```python
@helion.kernel(autotune_effort="none")
def fn(x: torch.Tensor, mode: hl.constexpr) -> torch.Tensor:
    if mode == "add":
        ...
    elif mode == "mul":
        ...

# Also useful for optional features:
def layer_norm_bwd(grad, x, weight, compute_bias_grad: hl.constexpr = True):
    if compute_bias_grad:
        grad_bias = ...  # Only compiled when True
```

## Static Range

`hl.static_range(n)` is a compile-time unrolled loop for small, known iteration counts:

```python
W = hl.specialize(w.shape[0])  # Must be specialized
for j in hl.static_range(W):
    v = hl.load(x, [tile.index + j], extra_mask=tile.index + j < N)
    acc += w[j] * v
```

## Atomic Operations

For concurrent writes (e.g., split-K matmul), use atomics:

```python
hl.atomic_add(out, [tile_m, tile_n], acc)
# Returns previous value:
old = hl.atomic_add(x, [i], y[i])
# With memory semantics:
hl.atomic_add(out, [tile_m, tile_n], acc, sem="acq_rel")
```

Also: `hl.atomic_max`, `hl.atomic_min`, `hl.atomic_and`, `hl.atomic_or`, `hl.atomic_xor`, `hl.atomic_xchg`, `hl.atomic_cas`.

Initialize the output with `torch.zeros` (not `torch.empty`) when using `atomic_add`.

## View Operations

Manipulate tile dimensions without copies:

```python
# Split last dim into two halves
lo, hi = hl.split(x[tile, :])

# Join two tiles along a new dimension
combined = hl.join(lo, hi)

# Reshape and permute work as expected
reshaped = acc.reshape([tile_m, 2, d // 2]).permute(0, 2, 1)
```

## Scan and Reduce Operations

```python
# Cumulative sum / product
result = hl.cumsum(x[tile, :], dim=1)
result = hl.cumprod(x[tile, :], dim=1)

# Custom associative scan
def combine_fn(left, right):
    return left + right
result = hl.associative_scan(combine_fn, x[tile, :], dim=1)

# Tuple associative scan (e.g., segment reduction)
def combine_fn(left_vals, left_ids, right_vals, right_ids):
    combined = torch.where(left_ids == right_ids, left_vals + right_vals, right_vals)
    return combined, right_ids
out_vals, _ = hl.associative_scan(combine_fn, (vals, ids), dim=0)

# Custom reduce
def max_combine(x, y):
    return torch.maximum(x, y)
result = hl.reduce(max_combine, x[tile, :], dim=1)

# Tuple reduce (e.g., argmax)
def argmax_combine(left, right):
    lv, li = left
    rv, ri = right
    take_right = rv > lv
    return torch.where(take_right, rv, lv), torch.where(take_right, ri, li)
max_val, max_idx = hl.reduce(argmax_combine, (values, indices), dim=1)
```

## Register Block Size and Tunables

For advanced control, register block sizes or tunable parameters that the autotuner explores:

```python
block_m = hl.register_block_size(m)
block_n = hl.register_block_size(n)
for tile_m in hl.tile(m, block_size=block_m):
    for tile_n in hl.tile(n, block_size=block_n):
        ...

# Custom tunable parameter with different fragment types
from helion.autotuner import PowerOfTwoFragment, IntegerFragment, EnumFragment

split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
multiplier = hl.register_tunable("multiplier", IntegerFragment(1, 10, 3))  # default=3
operation = hl.register_tunable("operation", EnumFragment(choices=(1, 2, 4)))
```

Tunables let the autotuner explore algorithmic choices — see the helion-perf skill for details on using `EnumFragment` for algorithmic branching.

## Closures and Epilogues

Functions captured from outer scope are automatically lifted to kernel arguments. This enables composable epilogue patterns:

```python
@helion.kernel(autotune_effort="none")
def matmul(x, y, epilogue=lambda acc, tile: acc) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))
    return out

# Usage: fused matmul + bias + relu
bias = torch.randn([n], device="cuda")
result = matmul(x, y, epilogue=lambda acc, tile: torch.relu(acc + bias[tile[1]]))
```

## Multiple Outputs

Kernels can return tuples. Allocate all outputs in host code before any tile loops:

```python
@helion.kernel(autotune_effort="none")
def layer_norm_fwd(x, weight, eps) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.size()
    out = torch.empty_like(x)
    mean = torch.empty([m], dtype=torch.float32, device=x.device)
    rstd = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        # ... compute
        out[tile_m, :] = normalized
        mean[tile_m] = mean_val
        rstd[tile_m] = rstd_val
    return out, mean, rstd
```

## Multiple Loop Blocks

A kernel can have multiple top-level `hl.tile` loops. Each becomes a separate grid launch within the same kernel. Use for forward+backward or multi-phase algorithms:

```python
@helion.kernel(config=helion.Config(block_sizes=[[32, 32], [32, 32]]))
def multi_phase(x, y) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    # Phase 1
    for tile_i, tile_j in hl.tile([m, n]):
        dx[tile_i, tile_j] = ...
    # Phase 2
    for tile_i2 in hl.tile(m):
        dy[tile_i2] = ...
    return dx, dy
```

Note: `block_sizes` uses a nested list when there are multiple loop blocks.

## Signal/Wait Synchronization

For cross-block synchronization (advanced):

```python
hl.signal(signal_pad, [tile], signal=1)
hl.wait(signal_pad, [tile], signal=1)

# With update value
hl.wait(signal_pad, [i], signal=1, update=2)
hl.signal(signal_pad, [i], signal=1, wait_for=0)
```

## Inline Triton and Assembly

When you need Triton-level or PTX-level control:

```python
# Inline Triton snippet — last line is the return value
result = hl.inline_triton(
    """
    tmp = {lhs} + {rhs}
    tmp
    """,
    args={"lhs": x_val, "rhs": y_val},
    output_like=x_val,
)

# Multiple outputs
sum_val, diff_val = hl.inline_triton(
    """
    s = {0} + {1}
    d = {0} - {1}
    s, d
    """,
    args=(a, b),
    output_like=(a, a),
)

# Side-effect only (no return value)
hl.inline_triton("tl.atomic_cas({0} + {1}, 0, 1)", args=(ptr, 0), output_like=None)

# Call an existing @triton.jit function by name or reference
result = hl.triton_kernel("my_triton_fn", args=(a, b), output_like=a)
result = hl.triton_kernel(my_triton_fn, args=(a, b), output_like=a)

# Inline PTX assembly
result = hl.inline_asm_elementwise(
    """
    {
        .reg .b64 ra, rb, rc;
        mov.b64 ra, { $2, $3 };
        mov.b64 rb, { $4, $5 };
        mul.f32x2 rc, ra, rb;
        mov.b64 { $0, $1 }, rc;
    }
    """,
    "=r,=r,r,r,r,r",  # LLVM constraint string
    [a, b],             # Input tensors
    dtype=torch.float32,# Output dtype
    is_pure=True,       # No side effects
    pack=2,             # Elements processed per invocation
)
```

## Host-Code Tensor Operations

Some operations on tensors in host code (before tile loops) trigger a `TensorOperationInWrapper` warning because they execute eagerly on the device. Common examples: `torch.nn.functional.pad`, `torch.broadcast_to`. Suppress with:

```python
@helion.kernel(
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def conv2d_kernel(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    b, h, w = x.size()
    kh, kw = k.size()[1:]
    x_padded = torch.nn.functional.pad(x, (0, kw, 0, kh, 0, 0), value=0.0)  # Host tensor op
    z = torch.empty_like(x)
    for i in hl.tile(b):
        for h_i in range(h):
            for w_i in range(w):
                patch = x_padded[i, h_i:h_i+kh, w_i:w_i+kw]
                z[i, h_i, w_i] = (patch * k[i, :, :]).sum(1).sum(1)
    return z
```

## Autograd Integration

Helion does not auto-generate backward passes. Write forward and backward kernels separately, then wrap with `torch.autograd.Function`:

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        result = my_fwd_kernel(x, weight)
        ctx.save_for_backward(x, weight)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x, grad_weight = my_bwd_kernel(grad_output, x, weight)
        return grad_x, grad_weight
```

## Common Pitfalls

- **Don't print inside kernels.** Use `hl.device_print` for debugging, or host-side logging.
- **Allocate outputs before tile loops.** All `torch.empty`/`torch.zeros` calls must be in host code.
- **Use float32 accumulators.** Even with float16 inputs, accumulate in float32 then cast: `out[tile] = acc.to(x.dtype)`.
- **Explicit indexing is mandatory.** Provide exactly `ndim` indices for every tensor access. `k[i]` on a 3D tensor fails — use `k[i, :, :]`.
- **Chain single-dim reductions.** Use `.sum(1).sum(1)` not `.sum([1, 2])`.
- **Don't use `:` on large dimensions.** `x[tile_m, :]` materializes the entire non-tiled dimension into registers. This is fine for small fixed dims (e.g. a 64-element head dim), but causes hangs or register spills for large dims (e.g. `hidden_dim=4096`). Instead, tile or iterate over that dimension too — use a sequential `for g in range(n_groups)` loop and load a bounded slice each iteration (see Tutorial Problem 10), or add a second `hl.tile` over the large dimension.
- **Default config is slow.** `autotune_effort="none"` is for correctness only. Don't benchmark with it.

## Reference Files

This skill includes reference files in the `references/` directory alongside this SKILL.md. Read them when you need deeper context:

### Documentation
- **`tutorials.md`** — 10 worked tutorials from first principles (vector add through flash attention). Read when implementing a new kernel pattern you haven't seen before.
- **`language_api.md`** — Complete API reference for `helion.language`. Read when you need exact function signatures, parameter types, or supported options for a specific API.

### Example Kernels (Pattern Reference)
Read these when implementing similar operations. Each file is a complete, runnable example with correctness checks.

- **`add.py`** — Simplest kernel: element-wise op with broadcasting, multi-dim tiling.
- **`softmax.py`** — Reductions: single-pass softmax, decomposed softmax, two-pass online softmax, backward pass. Shows `hl.full`, `register_block_size`, multi-pass inner loops.
- **`matmul.py`** — Matrix multiplication with epilogue closures, `torch.addmm`, forward+backward, `torch.autograd.Function` integration, multiple loop blocks.
- **`attention.py`** — Flash attention: online softmax, `torch.bmm`/`torch.baddbmm`, `hl.specialize`, reshape/transpose in host code, 3D tiling.
- **`layer_norm.py`** — Layer normalization forward+backward: `hl.constexpr` for optional bias, `register_block_size`, nested tile loops (`tile.begin`/`tile.end`), `hl.specialize`, multi-output tuple return.
- **`cross_entropy.py`** — Cross-entropy loss: `hl.load` with flat indexing, `ignore_warnings=[TensorOperationInWrapper]`, `view`-based index arithmetic.
- **`rms_norm.py`** — RMS normalization forward+backward: accumulator patterns, `hl.specialize`, `register_block_size`, `tile.id` for per-block partial results.
- **`segment_reduction.py`** — Associative scan: `hl.associative_scan` with tuple combine function, `hl.load` with `extra_mask`, `hl.atomic_add`, `tile.index`/`tile.block_size`.
