# TorchInductor Architecture Reference

## Overview

TorchInductor is PyTorch's native compiler backend, invoked by TorchDynamo. The pipeline:

```
TorchDynamo (FX graph capture)
    --> TorchInductor (lowering + codegen)
    --> Triton (GPU) or C++/OpenMP (CPU)
```

Entry point: `torch/_inductor/compile_fx.py`

## Define-by-Run IR

TorchInductor's IR (`torch/_inductor/ir.py`) is **not** a static data structure. IR nodes are
**Python callables** that accept SymPy expressions. The system analyzes and generates code by
swapping out `ops.*` implementations and re-executing the IR.

Example: `x.permute(1, 0) + x[2, :]` becomes a `Pointwise` object with an `inner_fn` that
takes symbolic loop indices, computes memory loads using SymPy indexing formulas, and returns
the result.

## Symbolic Shapes

- All non-trivial tensor dimensions are `sympy.Symbol` objects
- Specializes on 0 and 1, keeps everything else symbolic
- Memory operations use SymPy indexing formulas (`stride * index + offset`)
- Guards trigger recompilation when assumptions are violated
- `sizevars.py` handles symbolic size variable tracking

## Lowering Pipeline

```
1. FX graph (from Dynamo) arrives at compile_fx.py
2. Lowering (lowering.py): FX ops -> Inductor IR nodes (Pointwise, Reduction, etc.)
   - Each node has an `inner_fn` callable
3. FX passes (fx_passes/): Pattern matching rewrites
   - e.g., fusing decomposed softmax back into online-softmax
4. Scheduling (scheduler.py): Fusion decisions via can_fuse() / score_fusion()
   - Produces FusedSchedulerNode objects
5. Codegen (codegen/): Triton or C++ kernel generation
6. Output (output_code.py): Python wrapper launching generated kernels
```

## Reductions in TorchInductor

### IR Representation
- `SchedulerNode(ComputedBuffer)` with `ops.reduction()` calls
- Annotated with `ReductionHint` (e.g., `INNER`, `online_softmax_reduce`)

### Pattern Matching
- Decomposed patterns (e.g., `amax -> sub -> exp -> sum` for softmax) are recognized
  and rewritten into fused primitives like `prims.prepare_softmax_online`

### Fusion
- Multiple reduction/elementwise nodes fused into single `FusedSchedulerNode`
- For softmax: three nodes (two reductions + one elementwise) -> one kernel

### Persistent vs. Non-Persistent Reductions
- **Persistent** (reduction size <= 1024 for INNER): Loads entire reduction tile into
  registers in one shot. One global read, one global write. Fast but limited by register pressure.
- **Non-persistent/looped**: Streams tiles in a loop maintaining running state. Requires
  second pass to reload and normalize. Slower due to multiple global reads.

### Generated Triton Code Patterns
- Persistent: `tl.load` for full tile, then `triton_helpers.max2` / `tl.sum` with
  dimension-collapsing `[:, None]` broadcasts
- Non-persistent: explicit `for r0_offset in range(...)` loops with
  `triton_helpers.online_softmax_combine` and different cache eviction policies per pass

## Key Modules

| Module | Role |
|---|---|
| `ir.py` | IR node definitions (Pointwise, Reduction, etc.) |
| `lowering.py` | FX-to-IR lowering |
| `scheduler.py` | Fusion and scheduling |
| `codegen/` | Backend-specific code generation |
| `fx_passes/` | Graph rewrite passes (pattern matching) |
| `graph.py` | Computation graph construction |
| `loop_body.py` | Loop structure handling |
| `pattern_matcher.py` | Pattern-based optimization |
| `config.py` | Configuration knobs |
| `compile_fx.py` | Main compilation entry point |
| `sizevars.py` | Symbolic size variable tracking |

## Triton Code Generation

- Nearly all kernels auto-generated via Triton (not hand-written)
- Exceptions: matmul and conv use **templates** with auto-generated epilogue fusions
- `codegen/` subdirectory handles Triton emission
- `triton_bundler.py` bundles kernels
- Triton runtime compiles directly rather than using JIT caching

## Sources

- https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
- https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-Reduction/
- https://github.com/pytorch/pytorch/tree/main/torch/_inductor
- https://github.com/pytorch/pytorch/tree/main/torch/_dynamo/backends
