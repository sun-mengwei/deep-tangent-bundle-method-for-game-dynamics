"""Particle, score, and reference time integrators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from .dynamics import Diffusion, Game
from .projection import ParameterStructure, TangentProjection, evaluate_dtb_projection

if TYPE_CHECKING:
    from .experiment import ExperimentConfig

def dtb_step(
    theta: torch.Tensor,
    selected: torch.Tensor,
    particles: torch.Tensor,
    target_velocity: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    *,
    step_size: float,
    chunk_size: int,
    svd_rtol: float,
    tangent_inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, TangentProjection]:
    r"""Advance particles and parameters with the same projected increment.

    This performs

    ``X_{k+1} = X_k + h J_{S_k}(theta_k, X_k) alpha_k`` and
    ``theta_{k+1}[S_k] = theta_k[S_k] + h alpha_k``.

    The updated neural map is never evaluated to replace ``X_{k+1}``. There
    are no resets or refits. By default the tangent is evaluated at the
    current particles. Pass fixed ``tangent_inputs`` to evaluate the moving
    parameter basis at immutable reference labels instead.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    basis_inputs = particles if tangent_inputs is None else tangent_inputs
    if basis_inputs.shape != particles.shape:
        raise ValueError("tangent_inputs must have the particle shape")
    projection = evaluate_dtb_projection(
        theta,
        selected,
        basis_inputs,
        target_velocity,
        model,
        structure,
        chunk_size=chunk_size,
        svd_rtol=svd_rtol,
    )
    next_particles = (particles + step_size * projection.velocity).detach()
    next_theta = theta.detach().clone()
    next_theta[selected] += step_size * projection.alpha.detach()
    if not torch.isfinite(next_particles).all() or not torch.isfinite(next_theta).all():
        raise FloatingPointError("DTB update produced a nonfinite value")
    return next_theta, next_particles, projection


def advance_score(
    score: torch.Tensor,
    spatial_jacobian: torch.Tensor,
    gradient_divergence: torch.Tensor,
    *,
    step_size: float,
) -> torch.Tensor:
    r"""Euler-step the score transported by a velocity field.

    With ``q = grad(log rho)`` and velocity ``u``, this computes

    ``q_next = q - h ((D_x u)^T q + grad(div(u)))``.
    """

    if score.ndim != 2:
        raise ValueError("score must have shape (N, d)")
    expected = (score.shape[0], score.shape[1], score.shape[1])
    if spatial_jacobian.shape != expected:
        raise ValueError("score and spatial_jacobian shapes are inconsistent")
    if gradient_divergence.shape != score.shape:
        raise ValueError("gradient_divergence must have the score shape")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    transported = torch.einsum("nab,na->nb", spatial_jacobian, score)
    next_score = score - step_size * (transported + gradient_divergence)
    if not torch.isfinite(next_score).all():
        raise FloatingPointError("score update produced a nonfinite value")
    return next_score.detach()


def rk4_step(
    game: Game,
    particles: torch.Tensor,
    time_value: float,
    step_size: float,
) -> torch.Tensor:
    r"""Advance ``dX/dt = b(X,t)`` by one classical RK4 step.

    The update is ``X+ = X + h*(k1 + 2*k2 + 2*k3 + k4)/6`` with the standard
    four Runge--Kutta stages evaluated through ``game.velocity``.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    with torch.no_grad():
        first = game.velocity(particles, time_value)
        _validate_velocity(first, particles, "RK4 k1")
        second = game.velocity(
            particles + 0.5 * step_size * first,
            time_value + 0.5 * step_size,
        )
        _validate_velocity(second, particles, "RK4 k2")
        third = game.velocity(
            particles + 0.5 * step_size * second,
            time_value + 0.5 * step_size,
        )
        _validate_velocity(third, particles, "RK4 k3")
        fourth = game.velocity(
            particles + step_size * third,
            time_value + step_size,
        )
        _validate_velocity(fourth, particles, "RK4 k4")
        result = particles + (step_size / 6.0) * (
            first + 2.0 * second + 2.0 * third + fourth
        )
        if not torch.isfinite(result).all():
            raise FloatingPointError(f"RK4 state is nonfinite after t={time_value:g}")
    return result.detach()


def rk4_flow(
    game: Game,
    particles: torch.Tensor,
    time_value: float,
    interval: float,
    *,
    maximum_step: float,
) -> torch.Tensor:
    r"""Approximate the interval flow ``Phi_interval(particles)`` by RK4.

    Equal substeps no larger than ``maximum_step`` are used, with the final
    substep adjusted so the requested interval endpoint is reached exactly.
    """

    if interval <= 0 or maximum_step <= 0:
        raise ValueError("interval and maximum_step must be positive")
    substeps = max(1, int(np.ceil(interval / maximum_step - 1e-12)))
    substep_size = interval / substeps
    result = particles.detach().clone()
    for substep in range(substeps):
        result = rk4_step(
            game,
            result,
            time_value + substep * substep_size,
            substep_size,
        )
    return result


def _advance_reference(
    game: Game,
    diffusion: Diffusion | None,
    config: ExperimentConfig,
    particles: torch.Tensor,
    time_value: float,
    step_size: float,
    noise_generator: torch.Generator,
) -> torch.Tensor:
    """Advance the selected reference method, optionally with refined substeps."""

    maximum_step = config.reference_step_size
    if _reference_method(config) == "rk4":
        return rk4_flow(
            game,
            particles,
            time_value,
            step_size,
            maximum_step=step_size if maximum_step is None else maximum_step,
        )
    substep_count = (
        1
        if maximum_step is None
        else max(1, int(np.ceil(step_size / maximum_step - 1e-12)))
    )
    substep_size = step_size / substep_count
    next_particles = particles
    for substep in range(substep_count):
        next_particles = _advance_reference_one_step(
            game,
            diffusion,
            config,
            next_particles,
            time_value + substep * substep_size,
            substep_size,
            noise_generator,
        )
    return next_particles


def _advance_reference_one_step(
    game: Game,
    diffusion: Diffusion | None,
    config: ExperimentConfig,
    particles: torch.Tensor,
    time_value: float,
    step_size: float,
    noise_generator: torch.Generator,
) -> torch.Tensor:
    """Advance one selected reference-integrator substep."""

    with torch.no_grad():
        method = _reference_method(config)
        drift = game.velocity(particles, time_value)
        _validate_velocity(drift, particles, "reference drift")
        increment = step_size * drift
        if config.stochastic:
            if diffusion is None:
                raise RuntimeError("stochastic reference has no diffusion")
            noise_matrix = diffusion.noise_matrix(particles, time_value)
            if (
                noise_matrix.ndim != 3
                or noise_matrix.shape[0] != particles.shape[0]
                or noise_matrix.shape[1] != particles.shape[1]
                or noise_matrix.shape[2] < 1
            ):
                raise ValueError(
                    "diffusion noise matrix must have shape (N, d, brownian_dim)"
                )
            brownian = torch.randn(
                (particles.shape[0], noise_matrix.shape[2]),
                dtype=particles.dtype,
                generator=noise_generator,
            ).to(particles.device)
            increment = increment + step_size**0.5 * torch.einsum(
                "nir,nr->ni",
                noise_matrix,
                brownian,
            )
        next_particles = particles + increment
        if not torch.isfinite(next_particles).all():
            method = _reference_method(config)
            raise FloatingPointError(
                f"{method} state is nonfinite after t={time_value:g}"
            )
    return next_particles.detach()


def _reference_method(config: ExperimentConfig) -> str:
    if not config.run_reference:
        return "none"
    if config.reference_integrator != "auto":
        return config.reference_integrator
    return "euler_maruyama" if config.stochastic else "euler"


def _validate_velocity(
    velocity: torch.Tensor,
    particles: torch.Tensor,
    name: str,
) -> None:
    if velocity.shape != particles.shape:
        raise ValueError(f"{name} must have the particle shape")
    if not torch.isfinite(velocity).all():
        raise FloatingPointError(f"{name} contains a nonfinite value")

