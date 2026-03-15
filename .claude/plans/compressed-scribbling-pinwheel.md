# Plan: Add Reduction Op Support to Experimental Autograd

## Context

The experimental autograd (`helion/experimental/autodiff.py`) currently only supports elementwise kernels — it explicitly rejects any kernel with reduction operations. The goal is to iteratively add support for **simple inline reductions** (single `RootGraphInfo` with reduction ops like `sum`, `mean`, `amax`, `amin` inside the FX graph), while designing for future extension to multi-loop reduction patterns (`ReductionLoopGraphInfo`).

The core challenge: reductions change tensor shapes. The forward maps `[M, N] → [M]`, so the backward must map `grad_out [M] → grad_input [M, N]`. The current `GraphAnalyzer` and `FXToHelionConverter` assume all tensors share the same shape.

### Verified backward graph structures (from AOT Autograd)

**`sum(x, dim=-1)` backward:** `expand(tangents_1, [M, N])` — broadcast grad_out back to input shape.

**`amax(x, dim=-1)` backward:** `view(tangents_1, [M, 1])` → recompute `amax` → `eq(max, input)` → `sum(eq)` → `div` → `mul(div, eq)` — scatter gradient to max positions, splitting ties.

Both contain **concrete shape literals** (e.g., `[4, 1]`, `[4, 8]`) that must be made dynamic.

## Changes

### 1. Relax validation in `backward()` (autodiff.py:508-517)

Remove the `rolled_reductions` check entirely. Keep rejection of `ReductionLoopGraphInfo` and `ForLoopGraphInfo`. The single `RootGraphInfo` case should be allowed even with inline reductions.

```python
# REMOVE this block (lines 508-510):
if any(info.used_rdim for info in host_function.device_ir.rolled_reductions):
    raise exc.AutodiffNotSupported("reduction operations")

# KEEP the rest but allow single RootGraphInfo with reductions
```

### 2. Track output shape info in `GraphAnalyzer` (autodiff.py)

Add an `OutputMapping` dataclass:
```python
@dataclass
class OutputMapping:
    tensor_name: str
    fake_tensor: torch.Tensor | None
```

Change `extract_computation_graph` return type to also return `list[OutputMapping]` — one per stored output tensor. Populate from `node.meta["val"]` of stored values.

Also: strip Helion-internal metadata when copying computation nodes — only preserve `"val"` in `node.meta` to avoid confusing AOT Autograd.

### 3. Add `grad_out_shape` to `FXToHelionConverter` (autodiff.py)

Pass `grad_out_shape: tuple[int, ...]` into the converter. This determines:
- `grad_out` has fewer dims than the iteration space
- Load indexing for `grad_out` uses fewer tile indices

Change in `_build_source` (line 395):
```python
# Before: tensor_ndim = iter_ndim if p == "grad_out" else ...
# After:
tensor_ndim = len(self.grad_out_shape) if p == "grad_out" else len(self.tensor_shapes[p])
```

### 4. Handle concrete shapes in backward computation codegen (autodiff.py)

The backward graph from AOT Autograd contains `expand([M, N])` and `view([M, 1])` with hardcoded shapes. Add special handling in `_generate_computation`:

- **`expand`**: Match shape arg against known input tensor shapes → generate `expand_as(tensor_tile)` instead of `expand([4, 8])`
- **`view`/`reshape`**: Match shape arg → generate `view(tensor_tile.shape)` or infer the reshape from known shapes. For `view([M, 1])` patterns (unsqueeze-like), generate `.unsqueeze(-1)` or `.view(*tensor_tile.shape[:1], 1)`

Add helper `_find_tensor_by_shape(shape) -> str | None` that matches a concrete shape tuple against `self.tensor_shapes` and `self.grad_out_shape`.

For shapes that don't match any known tensor (like `[M, 1]`), use a heuristic: if the shape is the grad_out shape with a `1` appended, generate `unsqueeze(-1)`.

### 5. Plumb grad_out shape through `backward()` (autodiff.py)

```python
converter = FXToHelionConverter(
    backward_graph=backward_graph,
    input_mappings=input_mappings,
    input_tensors=inputs,
    grad_out_shape=tuple(grad_out.shape),  # NEW
)
```

### 6. Add tests (test/test_autodiff.py)

**Update `_check_backward` helper** to accept `input_shapes` and `grad_out_shape` parameters for non-uniform shapes.

**New test cases:**
- `test_sum_reduction` — `x[tile_m, :].sum(-1)`
- `test_mean_reduction` — `x[tile_m, :].mean(-1)`
- `test_amax_reduction` — `torch.amax(x[tile_m, :], dim=1)`
- `test_sum_mul_reduction` — `(x[tile_m, :] * y[tile_m, :]).sum(-1)` (composition)

**Update `test_error_reduction`** to test multi-loop reductions (still unsupported), not simple inline ones.

### 7. Update error message (helion/exc.py)

Refine `AutodiffNotSupported` to say "reduction operations with loop structure" or "multiple tile loops" instead of the generic "reduction operations".

## Files to Modify

| File | Changes |
|------|---------|
| `helion/experimental/autodiff.py` | Steps 1-5: validation, GraphAnalyzer, FXToHelionConverter, shape plumbing |
| `test/test_autodiff.py` | Step 6: new reduction tests, updated helper |
| `test/test_autodiff.expected` | Auto-generated via `EXPECTTEST_ACCEPT=1` |
| `helion/exc.py` | Step 7: refined error message |

## Implementation Order

1. **Step 1** (relax validation) + **Step 2** (GraphAnalyzer output mappings + metadata stripping)
2. **Steps 3-5** (FXToHelionConverter shape handling + grad_out_shape plumbing)
3. **Step 6** (tests) — iterate until `sum` passes, then `mean`, then `amax`
4. **Step 7** (error messages)

## Verification

```bash
# Run specific new tests
pytest test/test_autodiff.py -k "test_sum_reduction or test_mean_reduction" -x -vv -s

# Accept expected output
EXPECTTEST_ACCEPT=1 pytest test/test_autodiff.py -k "test_sum_reduction" -x -vv -s

# Run full autodiff suite to check no regressions
pytest test/test_autodiff.py -x -vv -s

# Verify generated backward code manually
# Use return_code=True to inspect the generated Helion and Triton code
```

## Extensibility Notes

This design extends to full graph support later:
- `OutputMapping` provides the shape-change tracking needed for multi-graph backward generation
- The `_find_tensor_by_shape` helper generalizes to any shape-changing op
- Multi-loop reductions (`ReductionLoopGraphInfo`) would require generating multiple tile loops in the backward kernel — the AST builder in `_build_source` can be extended with nested loops
- The validation gate in `backward()` can be progressively relaxed as more patterns are supported
