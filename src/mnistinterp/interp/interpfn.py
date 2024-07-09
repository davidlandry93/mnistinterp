"""The InterpFn is the trio of functions alpha, beta gamma as used in the Albergo2023
paper."""

import abc

import torch


class InterpFn(abc.ABC):
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def dalpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def dbeta(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def dgamma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(
        self, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return self.alpha(t) * x0 + self.beta(t) * x1 + self.gamma(t) * z


class LinearInterp(InterpFn):
    def __init__(self, noise_scale: float = 1.0):
        self.a = noise_scale

    def alpha(self, t):
        return 1.0 - t

    def beta(self, t):
        return t

    def gamma(self, t):
        return torch.sqrt(self.a * t * (1 - t))

    def dalpha(self, t):
        return torch.tensor(-1)

    def dbeta(self, t):
        return torch.tensor(1)

    def dgamma(self, t):
        return (self.a - 2 * self.a * t) / (2 * torch.sqrt(-self.a * (t - 1) * t))
