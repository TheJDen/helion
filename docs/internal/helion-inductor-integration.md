# Helion <-> TorchInductor Integration

## Architecture Overview

Helion is a Python DSL that compiles to Triton kernels. It integrates with PyTorch's
compilation pipeline at multiple levels through TorchInductor and TorchDynamo.

### Key Integration Files

```
helion/_compiler/
├── inductor_lowering.py          [1252 lines] Core IR lowering bridge
├── inductor_lowering_extra.py    [154 lines]  Custom lowering registrations
├── _inductor/
│   └── template_buffer.py        [445 lines]  torch.compile fusion
├── _dynamo/
│   ├── higher_order_ops.py       [346 lines]  HOP definitions & side table
│   └── variables.py              [413 lines]  Dynamo variable tracker
├── aten_lowering.py                           ATen op lowering registry
├── reduction_strategy.py                      Reduction codegen strategies
├── device_ir.py                               Graph info classes
└── generate_ast.py                            AST generation from Helion IR
```

## Inductor IR Lowering Pipeline

The core bridge is `helion/_compiler/inductor_lowering.py`. It intercepts Inductor's IR
and converts it to Helion/Triton code:

```
Inductor FX Graph
    | [prepare_graph_lowerings]
    v
Node Lowering Selection (APIFunc, Aten, SymInt)
    | [prepare_node_lowering]
    v
Graph Lowering Dispatch
    ├── APIFuncLowering (Helion's language operations)
    ├── AtenLowering (PyTorch ATen ops)
    └── SympyExprLowering (Symbolic expressions)
    |
    v
Inductor Buffers (ComputedBuffer with Pointwise/Reduction)
    ├── PointwiseLowering (elementwise ops)
    └── ReductionLowering (reduction ops)
    | [install_inductor_kernel_handlers]
    v
GenerateASTFromInductor (ops handler)
    |
    v
GraphInterpreter (run_node)
    |
    v
Triton AST
```

### Core Classes

- **`InductorLowering`** (base): Receives Inductor `ComputedBuffer`
- **`PointwiseLowering`**: Handles elementwise operations
- **`ReductionLowering`**: Handles sum, mean, amax, amin, etc.
- **`GenerateASTFromInductor`** (extends `DefaultHandler`): Converts Inductor IR to Triton AST
- **`GraphInterpreter`**: Interprets FX nodes as AST expressions

## torch.compile Fusion Path

When Helion kernels are called inside `torch.compile` with fusion enabled
(`_WIP_DEV_ONLY_HELION_TORCH_COMPILE_FUSION=1`):

### Step 1: Dynamo Tracing (`_dynamo/variables.py`)
- `HelionKernelVariable` tracker intercepts kernel calls
- Creates side table entries via `helion_kernel_side_table`
- Emits **Higher-Order Operator (HOP)** node into the FX graph

### Step 2: HOP Definitions (`_dynamo/higher_order_ops.py`)
- `helion_kernel_wrapper_mutation`: For kernels with mutations
- `helion_kernel_wrapper_functional`: For functional (pure) kernels
- Both implement proxy tracing, fake tensor generation, and functionalization
- Register with effect system: `EffectType.ORDERED`

### Step 3: Inductor Lowering (`_inductor/template_buffer.py`)
- `register_lowering(helion_kernel_wrapper_mutation)` creates `HelionTemplateBuffer`
- Extends `TritonTemplateBuffer` from Inductor
- Called by `lower_helion_kernel()` and `lower_helion_kernel_functional()`

### Step 4: Kernel Codegen & Rendering
- `HelionTemplateBuffer.render()`: Generates Triton source via `generate_ast()`
- `HelionTemplateBuffer.codegen_template_override()`: Inductor scheduler hook
- `HelionTemplateBuffer.emit_kernel_override()`: Emits imports and kernel definitions

## Config Patching

Helion patches Inductor config to preserve its own reduction semantics:

```python
# inductor_lowering.py: _patched_inductor_config()
patch = {
    "triton.codegen_upcast_to_fp32": True,   # FP32 math correctness
    "split_reductions": False,                # Keep reductions as Reduction IR
    "unroll_reductions_threshold": 1,         # Don't unroll into pointwise
    "use_fast_math": settings.fast_math,      # Optional fast math
}
```

## PyTorch Imports Used

| Component | Import | Purpose |
|-----------|--------|---------|
| `torch._inductor.ir.*` | TensorBox, ComputedBuffer, Reduction | IR representation |
| `torch._inductor.graph` | GraphLowering | Inductor's graph scheduler |
| `torch._inductor.lowering` | register_lowering | Dispatch table registration |
| `torch._inductor.virtualized` | V | Set/get graph, ops, kernel handlers |
| `torch._dynamo` | HigherOrderOperator, ProxyTorchDispatchMode | Tracing and fusion |
| `torch._functorch.aot_autograd` | aot_module_simplified | Gradient computation |
| `torch.fx` | Graph, Node, GraphModule | FX IR representation |

## Conceptual Mapping: Helion <-> Inductor

| Helion Concept | Inductor Analog |
|---|---|
| `hl.tile()` + tile indexing | Symbolic loop variables + SymPy indexing in `inner_fn` |
| `RootGraphInfo` (inline reduction) | `Reduction` IR node with `ReductionHint` |
| `ReductionLoopGraphInfo` | Multi-pass persistent/non-persistent reduction |
| Kernel fusion decisions | `scheduler.py` `can_fuse()` / `score_fusion()` |
| Python-as-IR (user-written) | Define-by-run IR (compiler-generated callables with `ops.*`) |
| `@helion.kernel` decorator | Inductor backend registration |
| `HelionTemplateBuffer` | `TritonTemplateBuffer` (for fusion) |

## Design Principles

1. **Single GPU kernel per Helion kernel** - Always compiles to exactly one Triton kernel
2. **Tile-centric programming** - Implicit tiling strategies and autotuning
3. **Automatic shape management** - Symbolic execution and dynamic shape support
4. **Inductor as backend** - Reuses Inductor's IR, scheduling, and lowering
5. **Seamless torch.compile integration** - HOPs + TemplateBuffer for fusion
6. **Extensible lowering** - Registry pattern for custom operations
