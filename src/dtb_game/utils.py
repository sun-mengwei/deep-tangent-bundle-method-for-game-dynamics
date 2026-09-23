"""Sampling, serialization, and plotting helpers for DTB experiments."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` and reject unavailable CUDA."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve the configured floating-point precision."""

    choices = {"float32": torch.float32, "float64": torch.float64}
    if name not in choices:
        raise ValueError("dtype must be 'float32' or 'float64'")
    return choices[name]


def warmup_cuda(device: torch.device, dtype: torch.dtype) -> None:
    """Initialize the CUDA context and cuBLAS handle on the main thread."""

    if device.type == "cuda":
        torch.cuda.init()
        value = torch.ones((2, 2), device=device, dtype=dtype)
        value @ value
        torch.cuda.synchronize()


def sample_initial_particles(
    count: int,
    dim: int,
    *,
    law: str,
    smoothing_std: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    uniform_low: float = 0.0,
    uniform_high: float = 1.0,
) -> torch.Tensor:
    """Draw reproducible CPU samples, then transfer the complete cloud."""

    particles, _ = sample_initial_with_score(
        count,
        dim,
        law=law,
        smoothing_std=smoothing_std,
        gaussian_mean=0.5,
        gaussian_std=0.15,
        dtype=dtype,
        device=device,
        generator=generator,
        uniform_low=uniform_low,
        uniform_high=uniform_high,
    )
    return particles


def sample_uniform_box(
    count: int,
    dim: int,
    *,
    low: float,
    high: float,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    r"""Draw ``z_i iid~Uniform([low, high]^dim)`` reproducibly.

    The samples are generated on CPU with a private seed and transferred as a
    complete tensor, so changing unrelated PyTorch random operations does not
    alter this Monte Carlo cloud.
    """

    return sample_initial_particles(
        count,
        dim,
        law="uniform",
        smoothing_std=1.0,
        dtype=dtype,
        device=device,
        generator=torch.Generator().manual_seed(int(seed)),
        uniform_low=low,
        uniform_high=high,
    )


def sample_initial_with_score(
    count: int,
    dim: int,
    *,
    law: str,
    smoothing_std: float,
    gaussian_mean: float,
    gaussian_std: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    uniform_low: float = 0.0,
    uniform_high: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an initial cloud and its analytical density score.

    ``smoothed_uniform`` is ``U([low,high]^d) + N(0, smoothing_std^2 I)``.
    Its score represents the softened boundary. ``uniform`` returns the
    interior score zero; its distributional boundary score is not represented.
    """

    if count < 1 or dim < 1:
        raise ValueError("count and dim must be positive")
    if smoothing_std <= 0 or gaussian_std <= 0:
        raise ValueError("smoothing_std and gaussian_std must be positive")
    if not math.isfinite(uniform_low) or not math.isfinite(uniform_high):
        raise ValueError("uniform interval bounds must be finite")
    if uniform_high <= uniform_low:
        raise ValueError("uniform_high must be greater than uniform_low")
    if law not in {"gaussian", "uniform", "smoothed_uniform"}:
        raise ValueError("law must be 'gaussian', 'uniform', or 'smoothed_uniform'")
    if law == "gaussian":
        noise = torch.randn((count, dim), dtype=dtype, generator=generator)
        particles = gaussian_mean + gaussian_std * noise
        score = -(particles - gaussian_mean) / gaussian_std**2
        return particles.to(device), score.to(device)

    particles = uniform_low + (uniform_high - uniform_low) * torch.rand(
        (count, dim), dtype=dtype, generator=generator
    )
    if law == "uniform":
        return particles.to(device), torch.zeros_like(particles, device=device)

    particles += smoothing_std * torch.randn(
        (count, dim),
        dtype=dtype,
        generator=generator,
    )
    upper = (particles - uniform_low) / smoothing_std
    lower = (particles - uniform_high) / smoothing_std
    density = (torch.special.ndtr(upper) - torch.special.ndtr(lower)).clamp_min(
        torch.finfo(dtype).tiny
    )
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    pdf_upper = inv_sqrt_2pi * torch.exp(-0.5 * upper.square())
    pdf_lower = inv_sqrt_2pi * torch.exp(-0.5 * lower.square())
    score = (pdf_upper - pdf_lower) / (smoothing_std * density)
    return particles.to(device), score.to(device)


def make_time_grid(final_time: float, step_size: float) -> np.ndarray:
    """Create a uniform grid and require final_time to be a step multiple."""

    if final_time <= 0 or step_size <= 0:
        raise ValueError("final_time and step_size must be positive")
    step_count = int(round(final_time / step_size))
    if not np.isclose(step_count * step_size, final_time, rtol=0.0, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of step_size")
    return np.linspace(0.0, final_time, step_count + 1)


def snapshot_indices(
    times: np.ndarray,
    requested: Sequence[float],
) -> dict[int, float]:
    """Map grid indices to requested snapshot times."""

    result: dict[int, float] = {}
    for value in sorted(set(float(time) for time in requested)):
        index = int(np.argmin(np.abs(times - value)))
        if not np.isclose(times[index], value, rtol=0.0, atol=1e-10):
            raise ValueError(f"snapshot time {value:g} is not on the time grid")
        result[index] = float(times[index])
    if 0 not in result:
        result[0] = float(times[0])
    if len(times) - 1 not in result:
        result[len(times) - 1] = float(times[-1])
    return dict(sorted(result.items()))


def to_numpy(value) -> np.ndarray:
    """Detach a tensor if needed and return a NumPy array."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def relative_l2_error(approximation: torch.Tensor, target: torch.Tensor) -> float:
    r"""Return ``||approximation-target||_2 / ||target||_2``."""

    if approximation.shape != target.shape:
        raise ValueError("approximation and target must have equal shapes")
    denominator = torch.linalg.vector_norm(target).clamp_min(
        torch.finfo(target.dtype).tiny
    )
    return float(
        (torch.linalg.vector_norm(approximation - target) / denominator).item()
    )


def paired_rms(first: torch.Tensor, second: torch.Tensor) -> float:
    r"""Return ``sqrt(mean_i ||first_i-second_i||_2^2)``."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired clouds must have equal matrix shapes")
    return float((first - second).square().sum(dim=1).mean().sqrt().item())


def write_csv(path: str | Path, columns, rows) -> Path:
    """Write a diagnostic table with one header row and return its path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    return target


def write_json(path: str | Path, data: Mapping[str, object]) -> None:
    """Write indented JSON, converting paths and NumPy scalars."""

    def default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot encode {type(value).__name__}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(data), indent=2, sort_keys=True, default=default) + "\n")










def sliced_wasserstein_distance(
    first,
    second,
    *,
    projections: int = 128,
    seed: int = 0,
) -> float:
    """Return a reproducible sliced 2-Wasserstein point-cloud distance."""

    first = to_numpy(first)
    second = to_numpy(second)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("point clouds must have the same (N, d) shape")
    if projections < 1:
        raise ValueError("projections must be positive")
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(projections, first.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    first_projection = np.sort(first @ directions.T, axis=0)
    second_projection = np.sort(second @ directions.T, axis=0)
    return float(
        np.mean(np.sqrt(np.mean((first_projection - second_projection) ** 2, axis=0)))
    )






