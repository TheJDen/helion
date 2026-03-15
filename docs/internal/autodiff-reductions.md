# Autodiff Reduction Support — Implementation Guide

## Overview

`helion.experimental.backward()` supports computing gradients for Helion kernels
with inline reduction operations (sum, mean, amax, amin) along a single dimension.
This document covers implementation details for future maintainers.

## Architecture

The backward pass generation pipeline:

1. **Validation** — Accept single `RootGraphInfo` (with optional `ReductionLoopGraphInfo`
   companions). Reject `ForLoopGraphInfo` (multi-tile-loop kernels) and `keepdim=True`.
2. **Graph extraction** (`GraphAnalyzer`) — Walk the forward FX graph, strip Helion
   internals, produce a pure PyTorch computation graph.
3. **Differentiation** (`differentiate_graph`) — Pass to AOT Autograd with full
   recomputation to get the backward FX graph.
4. **Code generation** (`FXToHelionConverter`) — Convert backward FX graph to a
   Helion kernel source string with shape-aware ops.
5. **Compilation** — Load the generated source as a module, compile and optionally
   autotune.

## Key Implementation Details

### Helion-internal op handling in graph extraction

The forward FX graph contains Helion-internal ops that AOT Autograd cannot process:

| Op | Handling |
|----|----------|
| `load` / `store` | Converted to placeholders/outputs (existing) |
| `_host_tensor` / `_get_symnode` | Skipped (infrastructure) |
| `_mask_to(tensor, fill_value)` | Identity pass-through; the backward kernel gets its own masking from the Helion compiler |
| `_inductor_lowering_extra` | Skipped, but its `args[0]` list is used to restore `None` args in ops that reference it via `_extra_args` |
| `_extra_args` kwarg | Stripped from computation nodes before copying |
| Unknown `_`-prefixed ops | Raise `AutodiffNotSupported` |

### Shape-aware backward code generation

AOT Autograd's backward graph contains concrete shape literals (e.g., `expand([64, 32])`,
`view([64, 1])`). The converter replaces these with dynamic expressions:

- **`unsqueeze(tensor, dim)`** → `tensor.view(tensor.shape[0], 1, ...)` — works
  around a Helion compiler bug where `unsqueeze(-1)` on 1D tiles transposes dimensions.
- **`expand(tensor, [M, N])`** → `tensor.expand_as(matched_input_tile)` — matches
  the concrete shape against known input tensor shapes.
- **`view(tensor, [M, 1])`** → `tensor.view(x_tile.shape[0], 1)` — matches dims
  against input/grad_out shapes, replacing concrete values with `tile.shape[i]`.

Shape matching uses `_find_tensor_by_shape()` which does exact tuple comparison against
`self.tensor_shapes` (concrete input shapes at generation time). The generated code
uses symbolic `tile.shape[i]` references that work at Helion compile time.

### Reduction-aware iteration

For reduction kernels (where `grad_out.ndim < input.ndim`), the backward kernel
iterates over `grad_out.shape` instead of `grad_x.shape`:

```python
# Generated for sum(x, dim=-1) backward:
for tile in hl.tile(grad_out.shape):     # 1D iteration over rows
    grad_out_tile = grad_out[tile]        # [BLOCK_SIZE]
    x_tile = x[tile, :]                   # [BLOCK_SIZE, N] — full row access
    ...
    grad_x[tile, :] = result              # store with full row access
```

This ensures reductions like `amax` in the recomputation see the full row, not a
tile subset. Without this, boundary tiles produce wrong gradients.

## Scope & Limitations

- **Single reduction dimension only** — `sum(dim=-1)` works, `sum(dim=[1, 2])` does not.
- **`keepdim=True` rejected** — Detected and raises `AutodiffNotSupported`.
- **`unsqueeze` workaround** — If Helion fixes `unsqueeze(-1)` lowering on 1D tiles,
  the `view` workaround becomes unnecessary but harmless.
- **Shape ambiguity** — `_find_tensor_by_shape` returns the first match when multiple
  inputs share a shape. Works for `expand_as` (shape-only) but could break for ops
  that need the correct tensor identity.

- **Bool reduction cast** — `sum(bool_tensor)` in Triton reduction loops causes
  a type mismatch (bool accumulator re-assigned to int64). The codegen detects
  bool inputs to sum/mean/prod and inserts `.to(torch.int64)` before the reduction.

## Testing

```bash
# Run all autodiff tests (requires GPU)
pytest test/test_autodiff.py -x -vv -s

# Run only reduction tests
pytest test/test_autodiff.py -k "reduction" -x -vv -s

# Accept new expected output after codegen changes
EXPECTTEST_ACCEPT=1 pytest test/test_autodiff.py -k "reduction" -x -vv -s

# Skip autotuning for faster iteration
HELION_USE_DEFAULT_CONFIG=1 pytest test/test_autodiff.py -x -vv -s

# Benchmark (shows default + autotuned columns)
python benchmarks/bench_autodiff_reductions.py
```

## Test matrix

| Test | Input shape | What it covers |
|------|-------------|----------------|
| `test_sum_reduction` | (64, 32) | Basic sum backward (expand) |
| `test_mean_reduction` | (64, 32) | Mean backward (_inductor_lowering_extra handling) |
| `test_amax_reduction` | (64, 32) | Amax backward (view, eq, recomputation) |
| `test_sum_mul_reduction` | (64, 32) × 2 | Multi-input reduction |
| `test_sum_reduction_boundary` | (65, 33) | Non-divisible sizes, masking |
| `test_amax_reduction_boundary` | (65, 33) | Recomputation masking at boundaries |
| `test_sum_reduction_large` | (1024, 512) | Larger sizes, tiling stress |
| `test_reduction_backward_autotune` | (128, 64) | Autotuned backward compiles correctly |
| `test_error_keepdim_reduction` | (64, 32) | keepdim=True rejected |
