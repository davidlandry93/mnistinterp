from typing import Callable
import abc

import tqdm
import torch
import torch.nn as nn

from .interpfn import InterpFn
from .lossfn import LossFn


class Solver(abc.ABC):
    def solve(
        self,
        model: nn.Module,
        interp_fn: InterpFn,
        loss_fn: LossFn,
        x0: torch.Tensor,
        steps: int | torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        raise NotImplementedError()


class EulerMaruyamaSolver(Solver):
    def __init__(
        self,
        epsilon: float | Callable[[float | torch.Tensor], float] = 1.0,
        only_denoise_from: float = 1.0,
    ):
        if isinstance(epsilon, (float, int)):
            self.epsilon_fn = lambda _: epsilon
        else:
            self.epsilon_fn = epsilon

        self.only_denoise_from = only_denoise_from

    def solve(
        self,
        model: nn.Module,
        interp_fn: InterpFn,
        loss_fn: LossFn,
        x0: torch.Tensor,
        steps: torch.Tensor | int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if isinstance(steps, int):
            steps = torch.linspace(0.0, 1.0, steps, device=x0.device)

        history = [x0]
        model_output_history = []

        for i, t in tqdm.tqdm(
            enumerate(steps[:-1]), desc="Generating", total=len(steps) - 1
        ):
            # # Make t broadcastable on state.
            # dims = tuple([1] * len(x0.shape))
            # t = t.reshape(dims)
            dt = steps[i + 1] - steps[i]

            xt = history[i]
            model_output = loss_fn.clamp(model(xt, t))
            drift = loss_fn.drift(xt, model_output, interp_fn, t)
            noise = loss_fn.denoiser(xt, model_output, interp_fn, t)

            epsilon_t = torch.tensor(self.epsilon_fn(t))

            if t > self.only_denoise_from:
                delta = -noise * dt
            elif i == len(steps) - 1:
                # Last step.
                delta = drift * dt - (epsilon_t / interp_fn.gamma(t)) * noise
            else:
                delta = (
                    drift * dt
                    - (epsilon_t / interp_fn.gamma(t)) * noise * dt
                    + torch.sqrt(2.0 * epsilon_t)
                    * torch.normal(torch.zeros_like(xt), torch.sqrt(dt))
                )

            history.append(xt + delta)
            model_output_history.append(model_output)

        return history, model_output_history


class HeunMaruyamaSolver(Solver):
    def __init__(self, epsilon: float | Callable[[float | torch.Tensor], float] = 1.0):
        if isinstance(epsilon, (float, int)):
            self.epsilon_fn = lambda _: epsilon
        else:
            self.epsilon_fn = epsilon

    def solve(
        self,
        model: nn.Module,
        interp_fn: InterpFn,
        loss_fn: LossFn,
        x0: torch.Tensor,
        steps: torch.Tensor | int,
    ):
        if isinstance(steps, int):
            steps = torch.linspace(0.0, 1.0, steps, device=x0.device)

        history = [x0]

        for i, t in enumerate(steps[:-1]):
            # # Make t broadcastable on state.
            # dims = tuple([1] * len(x0.shape))
            # t = t.reshape(dims)
            dt = steps[i + 1] - steps[i]

            xt = history[i]
            model_output = model(xt, t)
            drift = loss_fn.drift(xt, model_output, interp_fn, t)
            noise = loss_fn.denoiser(xt, model_output, interp_fn, t)

            epsilon_t = torch.tensor(self.epsilon_fn(t))

            dx_without_wiener = drift - (epsilon_t / interp_fn.gamma(t)) * noise
            wiener = torch.sqrt(2.0 * epsilon_t) * torch.normal(
                torch.zeros_like(xt), torch.sqrt(dt)
            )

            if i == len(steps) - 2:
                # Last step.
                step = dx_without_wiener * dt + wiener
            else:
                t_prime = t + dt
                epsilon_t_prime = self.epsilon_fn(t_prime)
                xt_prime = (xt + dx_without_wiener) * dt

                model_output_prime = model(xt_prime, t + dt)
                drift_prime = loss_fn.drift(xt, model_output_prime, interp_fn, t + dt)
                noise_prime = loss_fn.denoiser(
                    xt, model_output_prime, interp_fn, t + dt
                )

                dx_prime = drift_prime - (
                    epsilon_t_prime / interp_fn.gamma(t + dt) * noise_prime
                )

                dx = (dx + dx_prime) / 2

                step = dx * dt + wiener

            history.append(xt + step)
