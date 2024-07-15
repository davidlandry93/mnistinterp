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

    def denoiser(self, xt: torch.Tensor, model_output: torch.Tensor):
        raise NotImplementedError()


class DriftDenoiseLoss(LossFn):
    def __call__(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class VelocityDenoiseLoss(LossFn):
    def __call__(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class DriftScoreLoss(LossFn):
    def __call__(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class VelocityScoreLoss(LossFn):
    def __call__(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


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

    def denoiser(self, xt, model_output):
        z_hat = model_output[:, [2]]

        return z_hat


class X0ZLoss(LossFn):
    def __call__(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class X1ZLoss(LossFn):
    def __call__(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class XXLoss(LossFn):
    def __call__(self, nn_output, x0, x1, z, t, interp_fn: InterpFn) -> torch.Tensor:
        target = torch.concat([x0, x1], dim=1)

        return torch.square(nn_output - target)

    def clamp(self, nn_output) -> torch.Tensor:
        return torch.concat(
            [torch.clamp(nn_output[:, :2], -1.0, 1.0), nn_output[:, [2]]], dim=1
        )

    def target_names(self) -> tuple[str, str]:
        return ("x0", "x1")

    def drift(self, xt, model_output, interp_fn: InterpFn, t: torch.Tensor):
        x0_hat, x1_hat = (
            model_output[:, [0]],
            model_output[:, [1]],
        )

        z_hat = xt - interp_fn.alpha(t) * x0_hat - interp_fn.beta(t) * x1_hat

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
        z_hat = xt - interp_fn.alpha(t) * x0_hat - interp_fn.beta(t) * x1_hat

        return z_hat
