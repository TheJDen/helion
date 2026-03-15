---
name: helion-perf
description: Guide for tuning Helion GPU kernel performance, including autotuning, Config parameters, architecture targeting (Blackwell/Hopper/Ampere), indexing strategies, deployment, and benchmarking. Auto-activate when the user wants to optimize a Helion kernel, improve kernel throughput, autotune a kernel, target a specific GPU architecture, use FP8/NVFP4 quantized types, configure block sizes or num_warps, deploy tuned kernels to production, or benchmark kernel performance. Also activate when the user mentions Blackwell, Hopper, tensor descriptors, warp specialization, persistent kernels, or dynamic dispatch in the context of Helion.
---

# Helion Kernel Performance Tuning

You are optimizing GPU kernels written in Helion. This skill covers autotuning, Config parameters, architecture-specific targeting (especially Blackwell), dynamic dispatch, algorithmic branching via tunables, inline assembly, ACF files, and deployment patterns.

## Config vs Settings

Helion has two parameter types. Know the difference:

- **Config**: Controls GPU execution (block_sizes, num_warps, indexing, pid_type). Autotuned. Hardware-dependent.
- **Settings**: Controls compilation (autotune_effort, print_output_code, static_shapes). Never autotuned. Development-focused.

```python
@helion.kernel(
    # Settings (compilation control)
    static_shapes=True,
    autotune_effort="full",
    # Config (execution control) — usually let autotuner find this
    # config=helion.Config(block_sizes=[64, 64, 32], num_warps=8)
)
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    ...
```

## Autotuning

### Quick Iteration vs Production

| Effort | Command | Time | Use Case |
|--------|---------|------|----------|
| `"none"` | `HELION_AUTOTUNE_EFFORT=none` | Instant | Dev iteration, correctness checking |
| `"quick"` | `HELION_AUTOTUNE_EFFORT=quick` | ~3-10s | Fast dev tuning |
| `"full"` | `HELION_AUTOTUNE_EFFORT=full` | ~5-15 min | Production tuning |

During development, default to `autotune_effort="none"`. The default config is slow but correct — never benchmark with it. Only autotune when the user explicitly asks for performance numbers.

### Running Autotuning

```python
# Implicit: first call triggers autotuning
result = my_kernel(x, y)

# Explicit: separate tuning from execution
config = my_kernel.autotune((x, y), force=True)
config.save("configs/my_kernel.json")

# Multiple sizes
for tag, args in datasets.items():
    config = my_kernel.autotune(args)
    config.save(f"configs/my_kernel_{tag}.json")
```

### Direct Autotuner Control

```python
from helion.autotuner import LFBOTreeSearch

bound = my_kernel.bind((x, y))
tuner = LFBOTreeSearch(
    bound, (x, y),
    initial_population=200,   # Default 100
    copies=10,                # Default 5
    max_generations=40,       # Default 20
)
best = tuner.autotune()
```

Available autotuners: `LFBOTreeSearch` (default), `LFBOPatternSearch`, `DESurrogateHybrid`, `PatternSearch`, `DifferentialEvolutionSearch`, `FiniteSearch`, `RandomSearch`.

### Initial Population Strategies

- `from_random` (default for `"full"`): Full random exploration
- `from_default` (default for `"quick"`): Perturb around default config
- `from_best_available`: Seed from cached prior results — good for iterating on a kernel across sessions

```bash
HELION_AUTOTUNE_EFFORT=full HELION_AUTOTUNER_INITIAL_POPULATION=from_best_available python my_kernel.py
```

### Constraining Autotuning

Override specific config parameters while autotuning the rest:

```python
@helion.kernel(
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],  # Pin these
        "range_num_stages": [0, 0],      # Pin these
    }
    # block_sizes, num_warps, indexing, etc. still autotuned
)
def matmul(x, y):
    ...
```

Force persistent kernel strategies only:
```python
@helion.kernel(autotune_force_persistent=True)  # or HELION_AUTOTUNE_FORCE_PERSISTENT=1
```

## Algorithmic Branching with Tunables

The autotuner can search over discrete algorithmic choices using `EnumFragment`. This lets a single kernel definition express multiple algorithms, and the autotuner picks the best one for the hardware and input size.

```python
from helion.autotuner.config_fragment import EnumFragment

@helion.kernel()
def flexible_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)

    # Let the autotuner choose the algorithm
    algo = hl.register_tunable("algo", EnumFragment(choices=(1, 2, 3)))

    for tile_m in hl.tile(m):
        if algo == 1:
            # Single-pass: fast for small N
            out[tile_m, :] = torch.nn.functional.softmax(x[tile_m, :], dim=1)
        elif algo == 2:
            # Two-pass: better numerical stability
            vals = x[tile_m, :]
            amax = torch.amax(vals, dim=1, keepdim=True)
            exp = torch.exp(vals - amax)
            out[tile_m, :] = exp / torch.sum(exp, dim=1, keepdim=True)
        else:
            # Online softmax: memory-efficient for large N
            # ... multi-pass implementation
    return out
```

The autotuner evaluates each choice and selects the fastest. This is powerful for:
- Choosing between matmul algorithms (e.g., standard vs split-K)
- Selecting reduction strategies (tree vs sequential)
- Enabling/disabling vectorization tricks

Fragment types available:
- `EnumFragment(choices=(val1, val2, ...))` — discrete choices (first is default)
- `PowerOfTwoFragment(low, high)` — powers of 2 in range
- `IntegerFragment(low, high, default)` — integer range

## Config Parameters Reference

### Block Sizes

The most impactful parameter. Controls tile sizes for each `hl.tile` loop.

```python
# Single loop block
config=helion.Config(block_sizes=[64, 128])      # tile_m=64, tile_n=128

# With reduction dimension
config=helion.Config(block_sizes=[[64, 128], 32]) # [tile_m, tile_n]=grid, tile_k=32

# Multiple loop blocks (e.g., forward + backward phases)
config=helion.Config(block_sizes=[[32, 32], [32, 32]])
```

Guidelines:
- Larger tiles = fewer blocks, better data reuse, but more register pressure
- Matmul: try 64-256 for M/N, 16-64 for K
- Reductions: small outer tile (4-32), larger reduction tile (64-256)
- Powers of 2 are required

### Execution Resources

```python
helion.Config(
    num_warps=8,       # Warps per block (1-16). More = more parallelism, less registers per thread
    num_stages=4,      # Pipeline stages for software pipelining. More = better memory hiding
)
```

Rules of thumb:
- `num_warps=4` is a safe default. Use 8 for large matmuls. Use 1-2 for memory-bound kernels.
- `num_stages`: 2-4 for most kernels. Higher for matmuls with large K. Increases shared memory usage.

### Loop Orders

```python
helion.Config(
    loop_orders=[[1, 0]],  # Swap iteration order of first hl.tile
)
```

For matmul `hl.tile([m, n])`, `[0, 1]` iterates M-major, `[1, 0]` iterates N-major. The autotuner explores this.

### PID Type (Program ID Layout)

```python
helion.Config(
    pid_type="flat",                    # Standard linear (default)
    # pid_type="xyz",                   # 3D layout
    # pid_type="persistent_blocked",    # Persistent kernel, blocked distribution
    # pid_type="persistent_interleaved",# Persistent kernel, interleaved distribution
)
```

Persistent kernels keep thread blocks alive across multiple tiles. Better for small problems or when launch overhead matters. `persistent_interleaved` is preferred for Blackwell.

### L2 Grouping (Cache Locality)

```python
helion.Config(
    l2_groupings=[8],  # Group 8 blocks for L2 cache reuse (PID swizzling)
)
```

Reorders program IDs so spatially adjacent tiles execute on nearby SMs. Critical for matmul performance. Values of 4-32 are typical.

### Indexing Strategy

```python
helion.Config(
    indexing="pointer",            # Default. Address arithmetic. Always works.
    # indexing="block_ptr",        # Block pointer. Good for matmul patterns.
    # indexing="tensor_descriptor",# TMA-based. Best throughput on Hopper+/Blackwell.
)

# Per-operation indexing (order: loads first, then stores)
helion.Config(
    indexing=["pointer", "tensor_descriptor", "block_ptr"],  # 2 loads + 1 store
)
```

**When to use each:**
- `pointer`: Default fallback. Works everywhere.
- `block_ptr`: Better for regular tiled access (matmul). Good on Ampere+.
- `tensor_descriptor`: Best throughput for strided access on Hopper/Blackwell. Requires 16-byte aligned strides in last dimension. Falls back to `pointer` if constraints aren't met.

### Loop Optimization Parameters

```python
helion.Config(
    range_unroll_factors=[0, 1],       # Unroll factors for tl.range loops
    range_warp_specializes=[True, None],# Warp specialization per loop
    range_num_stages=[0, 3],           # Pipeline stages per loop
    range_multi_buffers=[None, False], # Accumulator multi-buffering
    range_flattens=[None, None],       # Loop flattening
    flatten_loops=[None],              # Flatten nested tile loops
)
```

For matmul: disable `range_unroll_factors` and `range_num_stages` since `tl.dot` is already pipelined via `num_stages`:
```python
@helion.kernel(
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    }
)
```

### Eviction Policies

```python
helion.Config(
    load_eviction_policies=["", "last"],  # Per-load site
)
```
- `""`: No policy (default)
- `"first"`: Evict from cache first (for streaming data read once)
- `"last"`: Keep in cache (for reused data)

## Architecture-Specific Targeting

### Blackwell (SM 10.0+, B200/B100)

Blackwell introduces several key features. When targeting Blackwell:

**1. Use `tensor_descriptor` indexing** — leverages TMA (Tensor Memory Accelerator):
```python
helion.Config(indexing="tensor_descriptor")
```

**2. Use persistent interleaved PID type** — better work distribution:
```python
helion.Config(pid_type="persistent_interleaved")
```

**3. Enable warp specialization** — separates producer/consumer warps for pipelining:
```python
helion.Config(
    range_warp_specializes=[True, None],  # Specialize outer loop
)
```

**4. Use `hl.dot` instead of `torch.addmm`** — gives direct control over matmul lowering:
```python
acc = hl.dot(q_tile, k_tile.T, out_dtype=torch.float32)
acc = hl.dot(p_fp8, v_tile, acc=acc)  # FP8 accumulation
```

**5. Non-transposed V in FP8** — Blackwell supports non-transposed V operand in FP8 dot:
```python
# Only on Blackwell: v_j doesn't need .T for FP8 dot
acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
```

**6. `hl.dot_scaled` for block-scaled matmul** — SM 10.0+ only:
```python
# Block-scaled FP8 matmul: each 32-element K block has one scale factor
# x_scale shape: [m, k // 32], y_scale shape: [n, k // 32]
acc = hl.dot_scaled(
    x[tile_m, :], x_scale[tile_m, :], "e4m3",    # format: "e2m1", "e4m3", "e5m2", "bf16", "fp16"
    y[:, tile_n], y_scale[tile_n, :], "e4m3",
    acc=acc,
)
```

**7. Vectorized PTX operations** — use inline asm for `f32x2` packed operations:
```python
def _mul_f32x2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=torch.float32,
        is_pure=True,
        pack=2,
    )
```

**8. Register pressure control** — tune via Triton config overrides:
```python
helion.Config(
    _triton_config_maxRegAutoWS=152,  # or 192
)
```

**9. Data partition factors** — control data distribution for warp-specialized loops:
```python
helion.Config(
    _triton_range_id_data_partition_factor=0,
    _triton_range_value_data_partition_factor=2,
)
```

**Complete Blackwell config example** (from `examples/blackwell_attention.py`):
```python
helion.Config(
    block_sizes=[256, 128],
    range_warp_specializes=[True, None],
    range_multi_buffers=[None, False],
    pid_type="persistent_interleaved",
    indexing="tensor_descriptor",
    num_warps=4,
    num_stages=3,
    _triton_range_id_data_partition_factor=0,
    _triton_range_value_data_partition_factor=2,
    _triton_config_maxRegAutoWS=152,
)
```

### Hopper (SM 9.0, H100/H200)

- `tensor_descriptor` indexing works (TMA support)
- Warp specialization supported
- `block_ptr` indexing is a good alternative
- ACF files available for fine-grained PTXAS control

### Ampere (SM 8.0, A100/A10G)

- Use `block_ptr` or `pointer` indexing (no TMA)
- No warp specialization
- `pid_type="flat"` or `"persistent_blocked"`

## Inline Triton and Assembly for Performance

When PyTorch ops don't map efficiently to the hardware, drop down to Triton or PTX:

```python
# Inline Triton — use for Triton-specific intrinsics
result = hl.inline_triton(
    """
    tmp = tl.math.fast_dividef({num}, {den})
    tmp
    """,
    args={"num": numerator, "den": denominator},
    output_like=numerator,
)

# Call a @triton.jit function — use for complex Triton logic
@triton.jit
def custom_op(a, b):
    return tl.math.rsqrt(a * a + b * b)

result = hl.triton_kernel(custom_op, args=(x_tile, y_tile), output_like=x_tile)

# Inline PTX assembly — use for hardware-specific instructions
# See examples/blackwell_attention.py for f32x2 packed multiply/FMA
result = hl.inline_asm_elementwise(
    asm_code,           # PTX assembly string
    constraints,        # LLVM constraint string (e.g., "=r,=r,r,r,r,r")
    [input_tensors],    # List of input tensors
    dtype=output_dtype, # Output element type
    is_pure=True,       # True if no side effects
    pack=2,             # Elements processed per invocation (for vectorized ops)
)
```

## FP8 and Quantized Kernels

### FP8 GEMM Pattern

```python
@helion.kernel(static_shapes=True, autotune_effort="none")
def fp8_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)  # Always FP32 accumulator
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(torch.float16)
    return out
```

### FP8 Attention

Key pattern: cast softmax probabilities to FP8 before the P@V dot:
```python
p = torch.exp2(qk_scaled)
p_fp8 = p.to(v.dtype)  # Cast to float8_e4m3fn
acc = hl.dot(p_fp8, v_tile, acc=acc)
```

### NVFP4 (E2M1) Quantization

See `examples/nvfp4_gemm.py` for the full pattern: load packed uint8, extract nibbles via bitwise ops, dequantize with piecewise formula, matmul with BF16 activations.

## Dynamic Dispatch and Deployment

### Key-Based Routing

Use `key=` to group inputs and re-select configs when the key changes:

```python
@helion.kernel(
    configs=[small_config, large_config],
    key=lambda x, y: helion.next_power_of_2(x.numel()),
    static_shapes=False,
)
def my_kernel(x, y):
    ...
```

This re-benchmarks provided configs when the power-of-2 bucket changes.

### Manual Routing with Compiled Configs

For full control, pre-compile configs and route at call time:

```python
bound = my_kernel.bind((example_x, example_y))

small_cfg = helion.Config.load("configs/small.json")
large_cfg = helion.Config.load("configs/large.json")

small_run = bound.compile_config(small_cfg)
large_run = bound.compile_config(large_cfg)

def routed_kernel(x, y):
    runner = small_run if x.numel() <= 2**16 else large_run
    return runner(x, y)
```

**Warning:** `kernel.bind()` specializes. With `static_shapes=True` (default), the bound kernel only works for the exact shape/stride of the example inputs. With `static_shapes=False`, it generalizes across shapes in the same bucket.

### Single Config Deployment

```python
best = helion.Config.load("configs/my_kernel.json")

@helion.kernel(config=best)
def my_kernel(x, y):
    ...
```

### Multiple Configs (lightweight selection)

```python
@helion.kernel(configs=[
    helion.Config.load("configs/small.json"),
    helion.Config.load("configs/large.json"),
], static_shapes=True)
def my_kernel(x, y):
    ...
```

Helion benchmarks each config on first call per specialization key and picks the fastest.

### Export Triton Source

Remove Helion from the serving path entirely:

```python
bound = my_kernel.bind((x, y))
triton_code = bound.to_triton_code(config)
# Deploy the Triton kernel directly, compile to PTX/cubins for Python-free execution
```

## Shape Specialization

`static_shapes=True` (default) bakes shapes into generated code — best performance but requires exact shape match.

`static_shapes=False` allows dynamic shapes with bucketed specialization (buckets: `{0, 1, >=2}` per dimension). Combine with `hl.specialize` for critical dimensions:

```python
@helion.kernel(static_shapes=False)
def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    hl.specialize(n)  # Hidden dim is compile-time constant, M is dynamic
    ...
```

Also available: `torch._dynamo.mark_static(tensor, dim)` for per-call specialization from outside the kernel.

## Benchmarking

```python
from triton.testing import do_bench

# Benchmark kernel
time_ms = do_bench(lambda: my_kernel(x, y))

# Compare against baseline
kernel_time = do_bench(lambda: my_kernel(x, y))
baseline_time = do_bench(lambda: torch.matmul(x, y))
print(f"Speedup: {baseline_time / kernel_time:.2f}x")

# TFLOPS for matmul
flops = 2 * m * n * k  # FLOPs for matmul
tflops = flops / (time_ms * 1e-3) / 1e12
```

## Debugging Performance

### See Generated Code

```python
@helion.kernel(print_output_code=True)  # or HELION_PRINT_OUTPUT_CODE=1
def my_kernel(...):
    ...
```

### Get Code Programmatically

```python
bound = my_kernel.bind((x, y))
triton_code = bound.to_triton_code(config)
print(triton_code)
```

### Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `HELION_PRINT_OUTPUT_CODE=1` | Print generated Triton code |
| `HELION_PRINT_REPRO=1` | Print standalone repro script |
| `HELION_AUTOTUNE_EFFORT=none` | Skip autotuning |
| `HELION_AUTOTUNE_EFFORT=quick` | Fast lightweight tuning |
| `HELION_FORCE_AUTOTUNE=1` | Force re-autotuning |
| `HELION_LOGS=all` | Enable INFO logging |
| `HELION_LOGS=+all` | Enable DEBUG logging |
| `HELION_DEBUG_DTYPE_ASSERTS=1` | Emit dtype assertions |
| `HELION_AUTOTUNE_LOG=mylog` | Write tuning telemetry to mylog.csv |
| `HELION_CACHE_DIR=/path` | Override autotuning cache directory |
| `HELION_SKIP_CACHE=1` | Ignore cached autotuning results |
| `HELION_AUTOTUNE_CONFIG_OVERRIDES='{"key": val}'` | Force config params during tuning |

### Common Performance Issues

1. **Default config is slow.** Always autotune before benchmarking. The default config exists for correctness, not speed.
2. **Wrong indexing strategy.** Try `tensor_descriptor` on Hopper+/Blackwell, `block_ptr` on Ampere.
3. **Accumulator dtype.** Always use `torch.float32` accumulators.
4. **L2 cache thrashing.** For matmul, ensure `l2_groupings` is set (4-32). The autotuner handles this.
5. **Too many/few warps.** Memory-bound kernels want fewer warps (1-2). Compute-bound want more (4-8).
6. **Reduction tile too small.** For softmax/layernorm, the reduction tile should be large enough to keep the GPU busy.
7. **Matmul pipeline interference.** For matmul kernels, disable `range_unroll_factors` and `range_num_stages` via `autotune_config_overrides` since `tl.dot` is pipelined by `num_stages`.

## Reference Files

This skill includes reference files in the `references/` directory alongside this SKILL.md. Read them when you need deeper context:

### Documentation
- **`config_reference.md`** — Complete `helion.Config` API reference with all parameter types and defaults. Read when you need exact parameter names, types, or valid ranges.
- **`deployment.md`** — Full deployment and autotuning guide: saving/loading configs, export to Triton source, key-based dispatch, `bind()`/`compile_config()` patterns. Read when deploying a tuned kernel.

### Example Kernels (Performance Reference)
Read these when optimizing similar operations. Each demonstrates advanced performance techniques.

- **`blackwell_attention.py`** — Full Blackwell-optimized flash attention: `tensor_descriptor` indexing, `persistent_interleaved` PID, warp specialization, `hl.dot` with FP8, `hl.inline_asm_elementwise` for `f32x2` vectorized multiply/FMA, `hl.split`/`hl.join` for subtiling, `EnumFragment` tunables, explicit config list, `_triton_*` overrides.
- **`fp8_gemm.py`** — FP8 GEMM: `hl.dot` with `float8_e4m3fn` inputs, FP32 accumulator, `static_shapes=True`, default config override for small block sizes.
- **`fp8_attention.py`** — FP8 attention: FP8 Q/K/V with `hl.dot`, softmax in FP32 then cast to FP8 for P@V, pre-transposed V layout, `hl.grid` for batch/head iteration.
- **`matmul_split_k.py`** — Split-K matmul: `hl.register_tunable` with `PowerOfTwoFragment`, `hl.atomic_add` for partial result accumulation, `torch.zeros` output (not `torch.empty`), conditional epilogue on `outer_k.begin == 0`.
