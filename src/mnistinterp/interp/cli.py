import hydra
import itertools
import logging
import math
import matplotlib.pyplot as plt
import mlflow
import mlflow.experiments
import omegaconf as oc
import torch.nn as nn
import torch.utils.data
import tqdm
from collections.abc import MutableMapping

from ..dataset import MNISTDataset
from .dataset import StochasticInterpolationDataset

from .interpfn import InterpFn
from .lossfn import LossFn
from .solver import Solver


APPROX_DATASET_LEN_TRAIN = 50_000
APPROX_DATASET_LEN_VAL = 10_000
N_SAMPLE_MODEL_OUTPUT_PLOT = 8


logger = logging.getLogger(__name__)


class TrainingState:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.step = 0
        self.val_loss_min = torch.inf


def get_experiment(cfg: oc.DictConfig):
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)

    if experiment is None:
        logger.info(
            f"Creating experiment {cfg.mlflow.experiment_name} because it did not exist."
        )
        experiment_id = mlflow.create_experiment(
            cfg.mlflow.experiment_name, artifact_location=cfg.mlflow.artifact_location
        )
        experiment = mlflow.get_experiment(experiment_id)

    return experiment


def flatten_config(dictionary: MutableMapping, parent_key="", separator=".") -> dict:
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_config(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def params_dict_for_logging(cfg: oc.DictConfig):
    params = flatten_config(dict(cfg))

    new_params = {}
    for k, v in params.items():
        if k.endswith("_target_"):
            new_params[k] = v.split(".")[-1]
        elif k.startswith("mlflow"):
            pass
        else:
            new_params[k] = v

    return new_params


def train(
    state: TrainingState,
    interp_fn: InterpFn,
    loss_fn: LossFn,
    solver: Solver,
    train_dataloader,
    val_dataloader,
    max_epochs: int,
    n_steps_per_epoch: int,
    device: torch.device,
    limit_train_batches=0,
    limit_val_batches=0,
    n_sampling_steps=5000,
    time_padding=1e-4,
):
    state.model.to(device)

    for epoch in range(max_epochs):
        mlflow.log_metric("epoch", epoch)
        training_loop(
            state,
            interp_fn,
            loss_fn,
            train_dataloader,
            n_steps_per_epoch,
            device,
            limit_train_batches=limit_train_batches,
        )
        validation_loop(
            state,
            interp_fn,
            loss_fn,
            solver,
            val_dataloader,
            device,
            limit_val_batches=limit_val_batches,
            n_sampling_steps=n_sampling_steps,
            time_padding=time_padding,
        )
        state.epoch += 1


def training_loop(
    state: TrainingState,
    interp_fn: InterpFn,
    loss_fn: LossFn,
    train_dataloader,
    n_steps_per_epoch,
    device: torch.device,
    limit_train_batches=0,
):

    state.model.train()

    total_loss = None
    n_samples = 0

    if limit_train_batches:
        dataloader = itertools.islice(train_dataloader, limit_train_batches)
        total = min(limit_train_batches, n_steps_per_epoch)
    else:
        dataloader = train_dataloader
        total = n_steps_per_epoch

    for batch in tqdm.tqdm(dataloader, total=total, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_size = batch["x0"].shape[0]

        state.optimizer.zero_grad()

        x0, x1, z, t = [batch[k] for k in ["x0", "x1", "z", "t"]]
        xt = interp_fn(x0, x1, z, t)
        model_output = state.model(xt, t)

        loss = loss_fn(model_output, x0, x1, z, t, interp_fn)
        mean_loss = loss.mean()
        mean_loss.backward()
        state.optimizer.step()
        state.scheduler.step()

        n_samples += batch_size
        if total_loss is None:
            total_loss = loss.sum(dim=0)
        else:
            total_loss += loss.sum(dim=0)

        if state.step % 50 == 0:
            mlflow.log_metric(
                "Train/loss_step", mean_loss.cpu().item(), step=state.step
            )

        state.step += 1

    if total_loss is None:
        raise RuntimeError("Train dataloader was empty.")

    mean_loss_per_channel = total_loss.mean(dim=[1, 2]) / n_samples
    mean_loss = mean_loss_per_channel.mean()
    mlflow.log_metric("Train/loss", mean_loss.item(), step=state.step)

    for i, target_label in enumerate(loss_fn.target_names()):
        mlflow.log_metric(
            f"Train/loss_{target_label}",
            mean_loss_per_channel[i].item(),
            step=state.step,
        )


def validation_loop(
    state: TrainingState,
    interp_fn: InterpFn,
    loss_fn: LossFn,
    solver: Solver,
    val_dataloader,
    device,
    limit_val_batches=0,
    n_sampling_steps=5000,
    time_padding=1e-4,
):
    state.model.eval()
    with torch.no_grad():
        total_loss = None
        n_samples = 0

        if limit_val_batches:
            dataloader = itertools.islice(val_dataloader, limit_val_batches)
        else:
            dataloader = val_dataloader

        for batch in tqdm.tqdm(dataloader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}

            x0, x1, z, t = [batch[k] for k in ["x0", "x1", "z", "t"]]
            xt = interp_fn(x0, x1, z, t)
            model_output = state.model(xt, t)

            loss = loss_fn(model_output, x0, x1, z, t, interp_fn)

            if total_loss is None:
                total_loss = loss.sum(dim=0)
            else:
                total_loss += loss.sum(dim=0)

            n_samples += loss.shape[0]

        if total_loss is None:
            raise RuntimeError("Validation dataloader was empty.")

        mean_loss_per_channel = total_loss.mean(dim=[1, 2]) / n_samples
        mean_loss = mean_loss_per_channel.mean()
        mlflow.log_metric("Val/loss", mean_loss.item(), step=state.step)

        if mean_loss_per_channel.mean() < state.val_loss_min:
            state.val_loss_min = mean_loss_per_channel.mean().item()

        for i, target_label in enumerate(loss_fn.target_names()):
            mlflow.log_metric(
                f"Val/loss_{target_label}",
                mean_loss_per_channel[i].item(),
                step=state.step,
            )

        if state.epoch % 1 == 0:
            plot_model_output(state, batch, interp_fn, loss_fn)
            plot_sampling(
                state,
                interp_fn,
                loss_fn,
                solver,
                val_dataloader,
                device,
                steps=n_sampling_steps,
                time_padding=time_padding,
            )


def plot_model_output(
    state: TrainingState, batch, interp_fn: InterpFn, loss_fn: LossFn
):
    x0, x1, z, t = [batch[k] for k in ["x0", "x1", "z", "t"]]
    xt = interp_fn(x0, x1, z, t)
    model_output = state.model(xt, t)

    model_output = model_output[:N_SAMPLE_MODEL_OUTPUT_PLOT]

    n_cols = 3 + loss_fn.n_targets

    fig, axs = plt.subplots(
        N_SAMPLE_MODEL_OUTPUT_PLOT, n_cols, sharex=True, sharey=True
    )
    fig.set_size_inches(12, 18)

    # Plot training data.
    for i in range(N_SAMPLE_MODEL_OUTPUT_PLOT):
        axs[i, 0].imshow(x0[i, 0].cpu().numpy())
        axs[i, 1].imshow(x1[i, 0].cpu().numpy())
        axs[i, 2].imshow(xt[i, 0].cpu().numpy())

        axs[i, 0].set_ylabel("{:.2f}".format(t[i].item()))

        if i == 0:
            axs[i, 0].set_title("x0")
            axs[i, 1].set_title("x1")
            axs[i, 2].set_title("xt")

    # Plot model output.
    for i in range(N_SAMPLE_MODEL_OUTPUT_PLOT):
        for j in range(loss_fn.n_targets):
            axs[i, j + 3].imshow(model_output[i, j].cpu().numpy())

    for i, label in enumerate(loss_fn.target_names()):
        axs[0, i + 3].set_title(label)

    # Remove all tick labels.
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    img_name = f"model_output_{state.epoch:03}.png"
    plt.savefig(img_name)
    mlflow.log_artifact(img_name)
    plt.close()


def plot_sampling(
    state: TrainingState,
    interp_fn: InterpFn,
    loss_fn: LossFn,
    solver: Solver,
    dataloader,
    device: torch.device,
    steps: int = 5000,
    time_padding: float = 1e-4,
):
    batch = next(iter(dataloader))

    x0 = batch["x0"][:N_SAMPLE_MODEL_OUTPUT_PLOT].to(device)

    ts = torch.linspace(time_padding, 1.0 - time_padding, steps, device=x0.device)
    xt_history, model_output_history = solver.solve(
        state.model, interp_fn, loss_fn, x0, ts
    )

    N_COLUMNS = 12
    column_idxs = (
        torch.linspace(0, len(xt_history) - 1, steps=N_COLUMNS).to(torch.int).numpy()
    )

    fig, axs = plt.subplots(
        N_SAMPLE_MODEL_OUTPUT_PLOT, N_COLUMNS, sharex=True, sharey=True
    )
    fig.set_size_inches(14, 10)

    for i in range(N_SAMPLE_MODEL_OUTPUT_PLOT):
        for j in range(N_COLUMNS):
            xt = xt_history[column_idxs[j]][i, 0].cpu().numpy()

            axs[i, j].imshow(xt)

            if i == 0:
                axs[i, j].set_title(f"t={ts[column_idxs[j]]:.2f}")

    # Remove ticks on all figures.
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    img_name = f"sampling_{state.epoch:03}.png"
    plt.savefig(img_name)
    plt.close()

    mlflow.log_artifact(img_name)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def cli(cfg: oc.DictConfig):
    n_steps_per_epoch = math.ceil(APPROX_DATASET_LEN_TRAIN / cfg.batch_size)
    cfg["n_steps_per_epoch"] = n_steps_per_epoch

    dataloaders = {}
    for subset in ["train", "val"]:
        origin_dataset = MNISTDataset(
            only_digit=cfg.origin_digit, subset=subset, pad32=True
        )
        destination_dataset = MNISTDataset(
            only_digit=cfg.destination_digit, subset=subset, pad32=True
        )

        if subset == "train":
            n_samples = cfg.n_steps_per_epoch * cfg.batch_size
        elif subset == "val":
            n_samples = APPROX_DATASET_LEN_VAL

        logger.info(
            f"Every epoch will train on {n_samples} samples in batches of {cfg.batch_size}."
        )
        dataset = StochasticInterpolationDataset(
            x0_dataset=origin_dataset,
            x1_dataset=destination_dataset,
            n_samples=n_samples,
        )

        dataloaders[subset] = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    interp_fn = hydra.utils.call(cfg.interpfn)
    loss_fn = hydra.utils.call(cfg.lossfn)

    model = hydra.utils.call(cfg.model, out_channels=loss_fn.n_targets)
    optimizer = hydra.utils.call(cfg.optimizer, model.parameters())
    scheduler = hydra.utils.call(cfg.scheduler.fn, optimizer)

    solver = hydra.utils.call(cfg.solver)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Will train on device: {device}")

    training_state = TrainingState(model, optimizer, scheduler)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow_experiment = get_experiment(cfg)
    mlflow.start_run(
        experiment_id=mlflow_experiment.experiment_id, run_name=cfg.mlflow.run_name
    )

    try:
        mlflow.log_params(params_dict_for_logging(cfg))
        train(
            training_state,
            interp_fn,
            loss_fn,
            solver,
            dataloaders["train"],
            dataloaders["val"],
            cfg.max_epochs,
            cfg.n_steps_per_epoch,
            device,
            limit_train_batches=cfg.limit_train_batches,
            limit_val_batches=cfg.limit_val_batches,
            n_sampling_steps=cfg.n_sampling_steps,
            time_padding=cfg.time_padding,
        )
    finally:
        mlflow.end_run()
