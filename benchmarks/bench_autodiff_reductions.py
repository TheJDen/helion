"""Benchmark Helion backward vs PyTorch autograd for reduction kernels."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

# --- Kernel definitions ---


@helion.kernel(autotune_effort="none")
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


@helion.kernel(autotune_effort="none")
def mean_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].mean(-1)
    return out


@helion.kernel(autotune_effort="none")
def amax_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amax(x[tile_m, :], dim=1)
    return out


@helion.kernel(autotune_effort="none")
def sum_mul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = (x[tile_m, :] * y[tile_m, :]).sum(-1)
    return out


# --- Benchmark helpers ---


def bench_fn(fn: Callable[[], None], warmup: int = 10, iters: int = 100) -> float:
    """Time a function using CUDA events. Returns median ms."""
    times: list[float] = []
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def _make_helion_bwd(
    helion_kernel: Callable[..., torch.Tensor],
    grad_out: torch.Tensor,
    inputs: list[torch.Tensor],
) -> Callable[[], None]:
    def fn() -> None:
        helion.experimental.backward(helion_kernel, grad_out, *inputs)

    return fn


def _make_pytorch_bwd(
    pytorch_fn: Callable[..., torch.Tensor],
    grad_out: torch.Tensor,
    inputs: list[torch.Tensor],
) -> Callable[[], None]:
    def fn() -> None:
        inputs_pt = [inp.detach().requires_grad_(True) for inp in inputs]
        out = pytorch_fn(*inputs_pt)
        out.backward(grad_out)

    return fn


def _warmup_and_bench(
    helion_kernel: Callable[..., torch.Tensor],
    grad_out: torch.Tensor,
    inputs: list[torch.Tensor],
    autotune: bool = False,
    autotune_effort: str | None = None,
) -> float | None:
    """Warm up (compile + optional autotune) and benchmark. Returns median ms or None on failure."""
    try:
        helion_kernel(*[inp.clone() for inp in inputs])
        helion.experimental.backward(
            helion_kernel,
            grad_out,
            *inputs,
            autotune=autotune,
            autotune_effort=autotune_effort,
        )
        return bench_fn(_make_helion_bwd(helion_kernel, grad_out, inputs))
    except Exception as e:
        print(f"    [autotune failed: {e!s:.60s}]", file=sys.stderr)
        return None


def bench_reduction(
    name: str,
    helion_kernel: Callable[..., torch.Tensor],
    pytorch_fn: Callable[..., torch.Tensor],
    shapes: list[tuple[int, int]],
    n_inputs: int = 1,
) -> None:
    """Benchmark a reduction kernel's backward pass (default + autotuned)."""
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")
    print(
        f"  {'Shape':>16s}  {'Default (ms)':>12s}  {'Tuned (ms)':>11s}"
        f"  {'PyTorch (ms)':>13s}  {'Def spdup':>9s}  {'Tune spdup':>10s}"
    )
    print(f"  {'-' * 16}  {'-' * 12}  {'-' * 11}  {'-' * 13}  {'-' * 9}  {'-' * 10}")

    for m, n in shapes:
        inputs = [
            torch.randn(m, n, device="cuda", dtype=torch.float32)
            for _ in range(n_inputs)
        ]
        grad_out = torch.randn(m, device="cuda", dtype=torch.float32)

        # Default config
        t_default = _warmup_and_bench(helion_kernel, grad_out, inputs)

        # Clear cached backward so autotuner runs fresh
        bound = helion_kernel.bind(tuple(inputs))
        bound._backward_compiled = None

        # Autotuned
        t_tuned = _warmup_and_bench(
            helion_kernel, grad_out, inputs, autotune=True, autotune_effort="quick"
        )

        # PyTorch reference
        t_pytorch = bench_fn(_make_pytorch_bwd(pytorch_fn, grad_out, inputs))

        def fmt(t: float | None) -> str:
            return f"{t:.4f}" if t is not None else "FAIL"

        def spd(t: float | None, ref: float) -> str:
            if t is None:
                return "N/A"
            return f"{ref / t:.2f}x"

        shape_str = f"({m}, {n})"
        print(
            f"  {shape_str:>16s}  {fmt(t_default):>12s}  {fmt(t_tuned):>11s}"
            f"  {t_pytorch:13.4f}  {spd(t_default, t_pytorch):>9s}  {spd(t_tuned, t_pytorch):>10s}"
        )


def main() -> None:
    shapes = [
        (1024, 512),
        (4096, 1024),
        (8192, 2048),
    ]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    bench_reduction("sum(x, dim=-1)", sum_kernel, lambda x: x.sum(-1), shapes)
    bench_reduction("mean(x, dim=-1)", mean_kernel, lambda x: x.mean(-1), shapes)
    bench_reduction(
        "amax(x, dim=-1)", amax_kernel, lambda x: torch.amax(x, dim=-1), shapes
    )
    bench_reduction(
        "(x * y).sum(-1)",
        sum_mul_kernel,
        lambda x, y: (x * y).sum(-1),
        shapes,
        n_inputs=2,
    )


if __name__ == "__main__":
    main()
