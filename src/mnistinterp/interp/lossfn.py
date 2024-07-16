"""The choice of loss functions used to train the neural network.

The reference material proposes four combination of loss functions to target, and 
more variants are introduced in subsequent material.
The main idea is that we should train the neural network to target 2 or 3 losses 
in a way that allows us to compute dX/dt during sampling."""

import abc

import torch


from .interpfn import InterpFn


class LossFn(abc.ABC):
    def __call__(
        self, nn_output: torch.Tensor, x0, x1, z, t, interpfn: InterpFn
    ) -> torch.Tensor:
        raise NotImplementedError()

    def clamp(self, nn_output: torch.Tensor) -> torch.Tensor:
        return nn_output

    @property
    def n_targets(self) -> int:
        return len(self.target_names())

    @abc.abstractmethod
    def target_names(self) -> tuple[str, ...]:
        raise NotImplementedError()

    def drift(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        interp_fn: InterpFn,
        t: torch.Tensor,
    ):
        raise NotImplementedError()

    def denoiser(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        interp_fn: InterpFn,
        t: torch.Tensor,
    ):
        raise NotImplementedError()

    def best_estimate(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        raise NotImplementedError()


class DriftScoreLoss(LossFn):
    def __call__(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class VelocityDenoiseLoss(LossFn):
    def __call__(self, nn_output, x0, x1, z, t, interp_fn: InterpFn) -> torch.Tensor:
        target_drift = interp_fn.velocity(x0, x1, t)
        target_noise = z

        target = torch.concat([target_drift, target_noise], dim=1)

        return torch.square(target - nn_output)

    def clamp(self, nn_output) -> torch.Tensor:
        return torch.stack(
            [torch.clamp(nn_output[:, 0], -1.0, 1.0), nn_output[:, 1]], dim=1
        )

    def target_names(self) -> tuple[str, str]:
        return ("velocity", "denoise")

    def drift(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        return model_output[:, [0]] - interp_fn.dgamma(t) * model_output[:, [1]]

    def denoiser(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        return model_output[:, [1]]

    def best_estimate(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        velocity, z_hat = model_output[:, 0], model_output[:, 1]

        delta = (1.0 - t).reshape(-1, 1, 1, 1)
        gamma = interp_fn.gamma(t).reshape(-1, 1, 1, 1)

        return xt + delta * velocity - gamma * z_hat


class DriftDenoiseLoss(LossFn):
    def __call__(self, nn_output, x0, x1, z, t, interp_fn: InterpFn) -> torch.Tensor:
        target_drift = interp_fn.drift(x0, x1, z, t)
        target_noise = z

        target = torch.concat([target_drift, target_noise], dim=1)

        return torch.square(target - nn_output)

    def clamp(self, nn_output) -> torch.Tensor:
        return nn_output

    def target_names(self) -> tuple[str, str]:
        return ("drift", "denoise")

    def drift(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        return model_output[:, [0]]

    def denoiser(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        return model_output[:, [1]]

    def best_estimate(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        delta = (1.0 - t).reshape(-1, 1, 1, 1)
        return xt + model_output[:, [0]] * delta


class VelocityScoreLoss(LossFn):
    def __call__(self, nn_output, x0, x1, z, t, interp_fn: InterpFn) -> torch.Tensor:
        target_drift = interp_fn.velocity(x0, x1, t)

        gamma = interp_fn.gamma(t).reshape(-1, 1, 1, 1)
        target_score = z / gamma

        target = torch.concat([target_drift, target_score], dim=1)

        return torch.square(target - nn_output)

    def clamp(self, nn_output) -> torch.Tensor:
        return torch.stack(
            [torch.clamp(nn_output[:, 0], -1.0, 1.0), nn_output[:, 1]], dim=1
        )

    def target_names(self) -> tuple[str, str]:
        return ("velocity", "score")

    def drift(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        gamma = interp_fn.gamma(t).reshape(-1, 1, 1, 1)
        dgamma = interp_fn.dgamma(t).reshape(-1, 1, 1, 1)

        return model_output[:, [0]] - gamma * dgamma * model_output[:, [1]]

    def denoiser(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        gamma = interp_fn.gamma(t).reshape(-1, 1, 1, 1)
        return model_output[:, [1]] / gamma

    def best_estimate(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        velocity, score_hat = model_output[:, 0], model_output[:, 1]

        delta = (1.0 - t).reshape(-1, 1, 1, 1)

        return xt + delta * velocity - score_hat


class XXZLoss(LossFn):
    def __call__(self, nn_output, x0, x1, z, t, interp_fn: InterpFn) -> torch.Tensor:
        target = torch.concat([x0, x1, z], dim=1)

        return torch.square(nn_output - target)

    def clamp(self, nn_output) -> torch.Tensor:
        return torch.concat(
            [torch.clamp(nn_output[:, :2], -1.0, 1.0), nn_output[:, [2]]], dim=1
        )

    def target_names(self) -> tuple[str, str, str]:
        return ("x0", "x1", "z")

    def drift(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        x0_hat, x1_hat, z_hat = (
            model_output[:, [0]],
            model_output[:, [1]],
            model_output[:, [2]],
        )

        drift = (
            interp_fn.dalpha(t) * x0_hat
            + interp_fn.dbeta(t) * x1_hat
            + interp_fn.dgamma(t) * z_hat
        )

        return drift

    def denoiser(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        z_hat = model_output[:, [2]]

        return z_hat

    def best_estimate(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        return model_output[:, [1]]


class X0ZLoss(LossFn):
    def __call__(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class X1ZLoss(LossFn):
    def __call__(self, nn_output, x0, x1, z, t, interp_fn: InterpFn) -> torch.Tensor:
        target = torch.concat([x1, z], dim=1)

        return torch.square(nn_output - target)

    def clamp(self, nn_output) -> torch.Tensor:
        return torch.stack(
            [torch.clamp(nn_output[:, 0], -1.0, 1.0), nn_output[:, 1]], dim=1
        )

    def target_names(self) -> tuple[str, str]:
        return ("x1", "z")

    def drift(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        z_hat = model_output[:, [1]]
        x1_hat = model_output[:, [0]]

        gamma = interp_fn.gamma(t).reshape(-1, 1, 1, 1)
        alpha = interp_fn.alpha(t).reshape(-1, 1, 1, 1)
        beta = interp_fn.beta(t).reshape(-1, 1, 1, 1)
        x0_hat = (xt - beta * x1_hat - gamma * z_hat) / alpha

        drift = (
            interp_fn.dalpha(t) * x0_hat
            + interp_fn.dbeta(t) * x1_hat
            + interp_fn.dgamma(t) * z_hat
        )

        return drift

    def denoiser(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        return model_output[:, [1]]

    def best_estimate(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        return model_output[:, [0]]


class XXLoss(LossFn):
    def __call__(self, nn_output, x0, x1, z, t, interp_fn: InterpFn) -> torch.Tensor:
        target = torch.concat([x0, x1], dim=1)

        return torch.square(nn_output - target)

    def clamp(self, nn_output) -> torch.Tensor:
        return torch.clamp(nn_output, -1.0, 1.0)

    def target_names(self) -> tuple[str, str]:
        return ("x0", "x1")

    def drift(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        x0_hat, x1_hat = (
            model_output[:, [0]],
            model_output[:, [1]],
        )

        gamma = interp_fn.gamma(t).reshape(-1, 1, 1, 1)
        z_hat = (xt - interp_fn.alpha(t) * x0_hat - interp_fn.beta(t) * x1_hat) / gamma

        drift = (
            interp_fn.dalpha(t) * x0_hat
            + interp_fn.dbeta(t) * x1_hat
            + interp_fn.dgamma(t) * z_hat
        )

        return drift

    def denoiser(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        x0_hat, x1_hat = (
            model_output[:, [0]],
            model_output[:, [1]],
        )
        gamma = interp_fn.gamma(t).reshape(-1, 1, 1, 1)
        z_hat = (xt - interp_fn.alpha(t) * x0_hat - interp_fn.beta(t) * x1_hat) / gamma

        return z_hat

    def best_estimate(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        return model_output[:, 1]
