import random

import torch
import torch.nn.functional as F
import torch.utils.data

from .mnist import mnist_test, mnist_train

VALIDATE_FROM = 4000
"""There are about 5000 examples per digit, so we can validate from
index 4000 to the end.
"""


class MNISTDataset:
    def __init__(self, only_digit=None, subset="train", pad32=True):
        dataset_fn = mnist_test if subset == "test" else mnist_train

        train_images, train_labels = dataset_fn()

        self.images = torch.from_numpy(train_images).float() / 255 - 0.5
        self.labels = torch.from_numpy(train_labels)

        mask = torch.ones(self.labels.shape[0], dtype=torch.bool)
        if only_digit is not None:
            mask = self.labels == only_digit

        if subset == "train":
            mask[VALIDATE_FROM:] = False
        if subset == "val":
            mask[:VALIDATE_FROM] = False

        self.images = self.images[mask]
        self.labels = self.labels[mask]

        if pad32:
            self.images = F.pad(self.images, [2, 2, 2, 2], value=-1.0)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        return {"image": self.images[idx], "label": self.labels[idx]}

    def __len__(self):
        return self.labels.shape[0]


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
        for _ in range(self.n_samples):
            x0 = self.x0_dataset[random.randint(0, len(self.x0_dataset) - 1)]["image"]
            x1 = self.x1_dataset[random.randint(0, len(self.x1_dataset) - 1)]["image"]
            z = torch.normal(mean=torch.zeros_like(x0))
            t = torch.rand(()) * (1 - 2 * self.time_padding) + self.time_padding

            # Add a fake channel dimension to make convolutions easier later.
            x0 = x0.unsqueeze(0)
            x1 = x1.unsqueeze(0)
            z = z.unsqueeze(0)

            yield {"x0": x0, "x1": x1, "z": z, "t": t}
