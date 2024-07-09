"""The choice of loss functions used to train the neural network.

The reference material proposes four combination of loss functions to target, and 
more variants are introduced in subsequent material.
The main idea is that we should train the neural network to target 2 or 3 losses 
in a way that allows us to compute dX/dt during sampling."""

import abc

import torch


class StochasticInterpolationLoss(abc.ABC):
    def loss(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def n_targets(self) -> int:
        raise NotImplementedError()


class DriftDenoiseLoss(StochasticInterpolationLoss):
    def loss(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def n_targets(self):
        return 2


class VelocityDenoiseLoss(StochasticInterpolationLoss):
    def loss(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def n_targets(self):
        return 2


class DriftScoreLoss(StochasticInterpolationLoss):
    def loss(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def n_targets(self):
        return 2


class VelocityScoreLoss(StochasticInterpolationLoss):
    def loss(self, x0, x1, z, t, nn_output: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def n_targets(self):
        return 2


class XAndNoiseLoss(StochasticInterpolationLoss):
    def loss(self, x0, x1, z, t nn_output: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def n_targets(self):
        return 3