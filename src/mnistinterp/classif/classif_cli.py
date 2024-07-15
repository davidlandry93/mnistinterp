import hydra
import omegaconf as oc
import matplotlib.pyplot as plt
import mlflow

import logging
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.utils.data
import tqdm
from einops import rearrange
import seaborn as sns
import pandas as pd


from ..dataset import MNISTDataset
from ..util import params_dict_for_logging, get_experiment

logger = logging.getLogger(__name__)


class MNISTClassifier(nn.Module):
    def __init__(self, embedding_size=512, dropout=0.2):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(32 * 32, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
        )

        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = rearrange(x, "b w h -> b (w h)")

        return self.classify(self.embed(x))


class TrainingState:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.epoch = 0
        self.step = 0
        self.val_loss_min = float("inf")

    def to_dict(self):
        return {
            'epoch': self.epoch,
            'step': self.step,
            'val_loss_min': self.val_loss_min,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load(self, checkpoint):
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.val_loss_min = checkpoint['val_loss_min']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

def train(
    state: TrainingState,
    train_dataloader,
    val_dataloader,
    max_epochs: int,
    device: torch.device,
):
    for _ in range(max_epochs):
        training_loop(state, train_dataloader, device=device)
        validation_loop(state, val_dataloader, device=device)
        state.epoch += 1


def training_loop(state: TrainingState, train_dataloader, device: torch.device):
    state.model.train()
    state.optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()

    score_acc = torch.tensor(0.0, device=device)
    n_samples = 0
    for i, batch in tqdm.tqdm(enumerate(train_dataloader), desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_size = batch["image"].shape[0]

        y_hat = state.model(batch["image"])

        cross_entropy = loss(y_hat, batch["label"])
        cross_entropy.backward()

        state.optimizer.step()
        state.step += 1

        score_acc += cross_entropy * batch_size
        n_samples += batch_size

        if i % 10 == 0:
            mlflow.log_metric("Train/loss_step", cross_entropy, step=state.step)

    mlflow.log_metric("Train/loss", score_acc.item() / n_samples, step=state.step)
    save_checkpoint(state, "latest")


def validation_loop(state: TrainingState, val_dataloader, device: torch.device):
    state.model.eval()
    loss = nn.CrossEntropyLoss()

    confusion = torch.zeros(10, 10, device=device)
    score_acc = torch.tensor(0.0, device=device)
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch["image"].shape[0]
            label = batch["label"].long()

            y_hat = state.model(batch["image"])

            cross_entropy = loss(y_hat, batch["label"])
            score_acc += cross_entropy * batch_size
            n_samples += batch_size

            label_hat = y_hat.argmax(dim=1)

            for digit in range(10):
                confusion[digit] += torch.bincount(
                    label_hat[label == digit], minlength=10
                )

        val_loss = score_acc.item() / n_samples

        if val_loss < state.val_loss_min:
            state.val_loss_min = val_loss
            mlflow.log_metric("Val/loss_min", val_loss, step=state.step)
            save_checkpoint(state, "best")

        mlflow.log_metrics(
            {
                "Val/loss": val_loss,
                "Val/precision": (
                    (torch.diag(confusion) / confusion.sum(dim=0)).mean().item()
                ),
                "Val/recall": (
                    (torch.diag(confusion) / confusion.sum(dim=1)).mean().item()
                ),
            },
            step=state.step,
        )

        if state.epoch % 5 == 0:
            plot_confusion_matrix(state, confusion)


def save_checkpoint(state: TrainingState, label: str):
    fname = f"{label}.ckpt"
    torch.save(state.to_dict(), fname)
    mlflow.log_artifact(fname)


def plot_confusion_matrix(state: TrainingState, confusion_matrix):
    rescaled_confusion = confusion_matrix / confusion_matrix.sum(dim=1, keepdims=True)

    confusion_df = pd.DataFrame(
        rescaled_confusion.cpu().numpy() * 100,
        index=list(range(10)),
        columns=list(range(10)),
    )

    sns.heatmap(confusion_df, annot=True, fmt=".1f")

    filename = f"confusion_matrix_{state.epoch:03}.png"
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def cli(cfg: oc.DictConfig):
    train_dataset = MNISTDataset(pad32=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )

    val_dataset = MNISTDataset(pad32=True, subset="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    model = hydra.utils.call(cfg.model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    state = TrainingState(model, optimizer)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Will train on device: {device}")
    model.to(device)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow_experiment = get_experiment(cfg)
    mlflow.start_run(
        experiment_id=mlflow_experiment.experiment_id, run_name=cfg.mlflow.run_name
    )


    try:
        end_status = "FINISHED"

        mlflow.log_params(params_dict_for_logging(cfg))
        mlflow.log_artifacts('.hydra')

        train(
            state,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            max_epochs=cfg.max_epochs,
            device=device,
        )
    except Exception as e:
        end_status = "FAILED"
        raise (e)
    finally:
        mlflow.end_run(end_status)
