import math
import random

import torch.utils.data

from ..dataset import MNISTDataset


class StochasticInterpolationDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        x0_dataset: MNISTDataset,
        x1_dataset: MNISTDataset,
        n_samples=5000,
        time_padding=1e-4,
    ):
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset
        self.n_samples = n_samples
        self.time_padding = time_padding

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            n_samples = self.n_samples / worker_info.num_workers

            if worker_info.id == 0:
                n_samples = math.ceil(n_samples)
            else:
                n_samples = math.floor(n_samples)
        else:
            n_samples = self.n_samples

        for _ in range(n_samples):
            x0 = self.x0_dataset[random.randint(0, len(self.x0_dataset) - 1)]["image"]
            x1 = self.x1_dataset[random.randint(0, len(self.x1_dataset) - 1)]["image"]
            z = torch.normal(mean=torch.zeros_like(x0))
            t = torch.rand(()) * (1 - 2 * self.time_padding) + self.time_padding

            # Add a fake channel dimension to make convolutions easier later.
            x0 = x0.unsqueeze(0)
            x1 = x1.unsqueeze(0)
            z = z.unsqueeze(0)

            yield {"x0": x0, "x1": x1, "z": z, "t": t}
