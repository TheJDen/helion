# Helion Autograd & Reductions (autodiff-reductions branch)

## Autograd Framework

**Location:** `helion/experimental/autodiff.py`

### Key Components

#### 1. GraphAnalyzer (lines 38-154)
- Analyzes forward Helion graphs extracted from FX traced kernels
- Identifies pure computation subgraphs by analyzing load/store patterns
- Tracks tensor names, placeholders, and computation nodes

#### 2. differentiate_graph() (lines 157-202)
- Uses PyTorch's AOT Autograd (`aot_module_simplified`) with full recomputation
- Generates backward FX graphs from forward computation
- Applies decompositions from `select_decomp_table()`
- Uses `min_cut_rematerialization_partition` strategy

#### 3. FXToHelionConverter (lines 205-541)
- Converts backward FX graphs to Helion kernel source code using AST
- Maps primal/tangent nodes to variable names
- Handles shape-changing operations (expand, view, reshape) with dynamic shapes
- Generates function with `@helion.kernel()` decorator
- Creates device loop for iteration over output shape

#### 4. backward() function (lines 544-680)
- Main entry point for computing gradients
- Validates kernel has been called at least once
- Extracts forward FX graph from bound kernel's host function
- Supports single `RootGraphInfo` (elementwise or inline reductions)
- Generates and caches backward kernel
- Returns gradients + optional Helion and Triton code

## Reduction Operations

### User-Facing API (`helion/language/reduce_ops.py`)

```python
hl.reduce(combine_fn, input_tensor, dim=None, other=None, keep_dims=False)
```

Or use standard PyTorch ops directly inside kernels:
```python
torch.sum(x, dim=-1)
torch.mean(x, dim=-1)
torch.amax(x, dim=-1)
torch.amin(x, dim=-1)
torch.prod(x, dim=-1)
```

### Combine Function Patterns

```python
# Sum reduction
def add_combine_fn(x, y): return x + y

# Max reduction
def max_combine_fn(x, y): return torch.maximum(x, y)

# Argmax (tuple format)
def argmax_combine_fn(left_tuple, right_tuple):
    left_value, left_index = left_tuple
    right_value, right_index = right_tuple
    take_right = right_value > left_value
    return (torch.where(take_right, right_value, left_value),
            torch.where(take_right, right_index, left_index))
```

### Compiler Pipeline for Reductions

**GraphInfo Hierarchy** (`device_ir.py`):
- `RootGraphInfo` - Single elementwise computation (supports inline reductions)
- `ForLoopGraphInfo` - Tiled loop structures
- `ReductionLoopGraphInfo` - Separate loop for reduction operations

**Reduction Strategies** (`reduction_strategy.py`):
- `PersistentReductionStrategy` - Persistent kernels with atomic accumulation
- `LoopedReductionStrategy` - Block-level reduction loop
- `BlockReductionStrategy` - Warp-level shuffles

**Code Generation:**
- Triton backend: Uses `tl.reduce()` with helper function
- CuTe backend: Maps to CuTe warp reduction operations
- Detects builtin patterns (sum, max, min, prod) for optimization

## Recent Changes (Commit 2c099fe8)

Added inline reduction support to `helion.experimental.backward()`:

1. **Relaxed validation** - Allow single `RootGraphInfo` with inline reductions
   (not just elementwise)
2. **OutputMapping** - Track forward output shapes for shape-changing reductions
3. **Metadata stripping** - Strip Helion-internal metadata before AOT Autograd
4. **Shape-changing ops** - Handle `expand`, `view` in backward codegen with dynamic shapes
5. **grad_out_shape** - Plumbed through `FXToHelionConverter` for correct broadcasting

## Current Limitations

| Supported | Not Supported |
|---|---|
| Single tile loop kernels | Multiple nested tile loops |
| Inline reductions (sum, mean, amax) | `ReductionLoopGraphInfo` (separate loop) |
| Elementwise + reduction combos | Complex multi-stage reductions |
| `@helion.kernel` with single `RootGraphInfo` | Multiple `RootGraphInfo` blocks |

## Test Coverage

**Files:**
- `test/test_autodiff.py` - 47 autograd tests
- `test/test_reduce.py` - Custom reduce tests
- `test/test_reductions.py` - Built-in reduction tests

**Key Test Cases:**
- `test_sum_reduction()` - Gradient through `x.sum(-1)`
- `test_mean_reduction()` - Gradient through `x.mean(-1)`
- `test_amax_reduction()` - Gradient through `torch.amax(x, dim=-1)`
- `test_sum_mul_reduction()` - Combined ops: `(x * y).sum(-1)`
- Error handling for unsupported patterns

**Test Helper:**
```python
_check_backward(kernel_fn, *inputs)
# Validates Helion backward against PyTorch autograd reference
```

## Useful Environment Variables

| Variable | Purpose |
|---|---|
| `HELION_LOGS=all` | Enable all logging |
| `HELION_PRINT_OUTPUT_CODE=1` | Print generated Triton code |
| `HELION_USE_DEFAULT_CONFIG=1` | Skip autotuning (fast iteration) |
| `HELION_DEBUG_DTYPE_ASSERTS=1` | Enable dtype checking |
