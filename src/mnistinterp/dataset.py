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

        self.images = (torch.from_numpy(train_images.copy()).float() / 255) * 2 - 1.0
        self.labels = torch.from_numpy(train_labels.copy())

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
