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


@helion.kernel(autotune_effort="none")
def matmul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    c = torch.zeros([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        c[tile_m, tile_n] = acc
    return c


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
    flops_per_element: int = 1,
) -> None:
    """Benchmark a reduction kernel's backward pass.

    Args:
        flops_per_element: FLOPs per input element in the backward pass.
            Used to compute TFLOPS throughput.
    """
    print(f"\n{'=' * 90}")
    print(f"  {name}")
    print(f"{'=' * 90}")
    print(
        f"  {'Shape':>16s}  {'Helion (ms)':>11s}  {'PyTorch (ms)':>12s}"
        f"  {'Speedup':>8s}  {'Helion TF/s':>11s}  {'PyTorch TF/s':>12s}"
    )
    print(f"  {'-' * 16}  {'-' * 11}  {'-' * 12}  {'-' * 8}  {'-' * 11}  {'-' * 12}")

    for m, n in shapes:
        inputs = [
            torch.randn(m, n, device="cuda", dtype=torch.float32)
            for _ in range(n_inputs)
        ]
        grad_out = torch.randn(m, device="cuda", dtype=torch.float32)

        # Default config
        t_helion = _warmup_and_bench(helion_kernel, grad_out, inputs)

        # PyTorch reference
        t_pytorch = bench_fn(_make_pytorch_bwd(pytorch_fn, grad_out, inputs))

        # Compute TFLOPS: total FLOPs across all inputs
        total_elements = m * n * n_inputs
        total_flops = total_elements * flops_per_element

        def tflops(t_ms: float | None, flops: int = total_flops) -> str:
            if t_ms is None or t_ms == 0:
                return "N/A"
            return f"{flops / (t_ms * 1e-3) / 1e12:.3f}"

        def fmt(t: float | None) -> str:
            return f"{t:.4f}" if t is not None else "FAIL"

        def spd(t: float | None, ref: float) -> str:
            if t is None:
                return "N/A"
            return f"{ref / t:.2f}x"

        shape_str = f"({m}, {n})"
        print(
            f"  {shape_str:>16s}  {fmt(t_helion):>11s}  {t_pytorch:12.4f}"
            f"  {spd(t_helion, t_pytorch):>8s}  {tflops(t_helion):>11s}  {tflops(t_pytorch):>12s}"
        )


def bench_matmul_backward(
    name: str,
    helion_kernel: Callable[..., torch.Tensor],
    pytorch_fn: Callable[..., torch.Tensor],
    shapes: list[tuple[int, int, int]],
) -> None:
    """Benchmark a matmul kernel's backward pass."""
    print(f"\n{'=' * 90}")
    print(f"  {name}")
    print(f"{'=' * 90}")
    print(
        f"  {'Shape':>24s}  {'Helion (ms)':>11s}  {'PyTorch (ms)':>12s}"
        f"  {'Speedup':>8s}  {'Helion TF/s':>11s}  {'PyTorch TF/s':>12s}"
    )
    print(f"  {'-' * 24}  {'-' * 11}  {'-' * 12}  {'-' * 8}  {'-' * 11}  {'-' * 12}")

    for m, k, n in shapes:
        a = torch.randn(m, k, device="cuda", dtype=torch.float32)
        b = torch.randn(k, n, device="cuda", dtype=torch.float32)
        grad_out = torch.randn(m, n, device="cuda", dtype=torch.float32)
        inputs = [a, b]

        t_helion = _warmup_and_bench(helion_kernel, grad_out, inputs)

        t_pytorch = bench_fn(_make_pytorch_bwd(pytorch_fn, grad_out, inputs))

        # Matmul backward: 2 matmuls, each 2*M*K*N FLOPs
        total_flops = 2 * 2 * m * k * n

        def tflops(t_ms: float | None, flops: int = total_flops) -> str:
            if t_ms is None or t_ms == 0:
                return "N/A"
            return f"{flops / (t_ms * 1e-3) / 1e12:.3f}"

        def fmt(t: float | None) -> str:
            return f"{t:.4f}" if t is not None else "FAIL"

        def spd(t: float | None, ref: float) -> str:
            if t is None:
                return "N/A"
            return f"{ref / t:.2f}x"

        shape_str = f"({m}, {k}) x ({k}, {n})"
        print(
            f"  {shape_str:>24s}  {fmt(t_helion):>11s}  {t_pytorch:12.4f}"
            f"  {spd(t_helion, t_pytorch):>8s}  {tflops(t_helion):>11s}  {tflops(t_pytorch):>12s}"
        )


def main() -> None:
    shapes = [
        (1024, 512),
        (4096, 1024),
        (8192, 2048),
    ]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    # flops_per_element: approximate FLOPs per input element in the backward
    #   sum:  expand (1 copy)
    #   mean: expand + div (2)
    #   amax: recompute max + eq + sum(eq) + div + mul (5)
    #   sum_mul: expand + mul per input (2)
    bench_reduction(
        "sum(x, dim=-1)", sum_kernel, lambda x: x.sum(-1), shapes, flops_per_element=1
    )
    bench_reduction(
        "mean(x, dim=-1)",
        mean_kernel,
        lambda x: x.mean(-1),
        shapes,
        flops_per_element=2,
    )
    bench_reduction(
        "amax(x, dim=-1)",
        amax_kernel,
        lambda x: torch.amax(x, dim=-1),
        shapes,
        flops_per_element=5,
    )
    bench_reduction(
        "(x * y).sum(-1)",
        sum_mul_kernel,
        lambda x, y: (x * y).sum(-1),
        shapes,
        n_inputs=2,
        flops_per_element=2,
    )

    matmul_shapes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    bench_matmul_backward(
        "matmul backward: grad_a = grad_out @ b.T, grad_b = a.T @ grad_out",
        matmul_kernel,
        torch.matmul,
        matmul_shapes,
    )


if __name__ == "__main__":
    main()
