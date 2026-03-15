"""Benchmark Helion backward vs PyTorch autograd for reduction kernels."""

from __future__ import annotations

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


def bench_reduction(
    name: str,
    helion_kernel: Callable[..., torch.Tensor],
    pytorch_fn: Callable[..., torch.Tensor],
    shapes: list[tuple[int, int]],
    n_inputs: int = 1,
) -> None:
    """Benchmark a reduction kernel's backward pass."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(
        f"  {'Shape':>20s}  {'Helion (ms)':>12s}  {'PyTorch (ms)':>13s}  {'Speedup':>8s}"
    )
    print(f"  {'-' * 20}  {'-' * 12}  {'-' * 13}  {'-' * 8}")

    for m, n in shapes:
        inputs = [
            torch.randn(m, n, device="cuda", dtype=torch.float32)
            for _ in range(n_inputs)
        ]
        grad_out = torch.randn(m, device="cuda", dtype=torch.float32)

        # Warm up Helion (triggers compilation)
        helion_kernel(*[inp.clone() for inp in inputs])
        helion.experimental.backward(helion_kernel, grad_out, *inputs)

        t_helion = bench_fn(_make_helion_bwd(helion_kernel, grad_out, inputs))
        t_pytorch = bench_fn(_make_pytorch_bwd(pytorch_fn, grad_out, inputs))

        speedup = t_pytorch / t_helion if t_helion > 0 else float("inf")
        print(
            f"  {f'({m}, {n})':>20s}  {t_helion:12.4f}  {t_pytorch:13.4f}  {speedup:7.2f}x"
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
