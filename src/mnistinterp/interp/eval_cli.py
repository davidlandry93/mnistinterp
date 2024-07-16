import random
import hydra
import logging
import numpy as np
import tqdm
import omegaconf as oc
import itertools
import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
import torch.utils.data
import pathlib
from ..dataset import MNISTDataset

from .lossfn import LossFn
from .interpfn import InterpFn
from .solver import Solver

from ..util import get_experiment, params_dict_for_logging

logger = logging.getLogger(__name__)


def load_config_from_id(run_id: str) -> oc.DictConfig:
    mlflow_run = mlflow.get_run(run_id)
    artifact_path = mlflow_run.info.artifact_uri

    if artifact_path is None:
        raise RuntimeError()

    artifact_path = pathlib.Path(artifact_path)

    run_cfg = oc.OmegaConf.load(artifact_path / "config.yaml")

    if not isinstance(run_cfg, oc.DictConfig):
        raise RuntimeError("Unexpected config format.")

    return run_cfg


def load_model_from_id(run_id: str, **kwargs) -> nn.Module:
    mlflow_run = mlflow.get_run(run_id)
    artifact_path = mlflow_run.info.artifact_uri

    if artifact_path is None:
        raise RuntimeError()

    artifact_path = pathlib.Path(artifact_path)

    run_cfg = oc.OmegaConf.load(artifact_path / "config.yaml")

    checkpoint = torch.load(artifact_path / "best.ckpt")
    model = hydra.utils.call(run_cfg.model, **kwargs)
    model.load_state_dict(checkpoint["model"])

    return model


@hydra.main(config_path="config", config_name="eval", version_base="1.3")
def eval_cli(cfg: oc.DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment = get_experiment(cfg)

    mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=cfg.mlflow.run_name
    )

    try:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"Will run evaluation onf device: {device}")

        run_cfg = load_config_from_id(cfg.run_id)

        # TEMPORARY FIX
        run_cfg["n_steps_per_epoch"] = 1

        loss_fn: LossFn = hydra.utils.call(run_cfg.lossfn)
        model = load_model_from_id(cfg.run_id, out_channels=loss_fn.n_targets)
        classifier = load_model_from_id(cfg.classifier_id)

        mlflow.log_params(
            {
                **params_dict_for_logging(run_cfg),
                **params_dict_for_logging(cfg),
            }
        )

        model.to(device)
        classifier.to(device)

        model.eval()
        classifier.eval()

        destination_dataset = MNISTDataset(
            only_digit=run_cfg.destination_digit, subset="test"
        )
        test_loader = torch.utils.data.DataLoader(
            destination_dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size
        )

        origin_dataset = MNISTDataset(only_digit=run_cfg.origin_digit, subset="test")
        origin_loader = torch.utils.data.DataLoader(
            origin_dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size
        )

        interp_fn: InterpFn = hydra.utils.call(run_cfg.interpfn)
        solver: Solver = hydra.utils.call(cfg.solver)

        with torch.no_grad():
            descriptors = []
            classifications = torch.zeros(10, device=device)

            for batch in tqdm.tqdm(test_loader, desc="Getting descriptors"):
                img = batch["image"].to(device)
                descriptor = classifier.embed(img)
                descriptors.append(descriptor)

                y_hat = classifier.classify(descriptor)
                label_hat = torch.argmax(y_hat, dim=1)
                classifications += torch.bincount(label_hat, minlength=10)

            precision_ref = (
                classifications[run_cfg.destination_digit] / classifications.sum()
            )
            mlflow.log_metric("ClassifierPrecisionRef", precision_ref.item())

            descriptors = torch.concat(descriptors, dim=0)

            ref_mu = descriptors.mean(dim=0)
            ref_cov = (
                torch.matmul((descriptors - ref_mu).T, (descriptors - ref_mu))
                / descriptors.shape[0]
            )

            x0s = []
            generated = []
            best_estimates = []
            path_lengths = []
            n_samples = []
            for batch in tqdm.tqdm(
                itertools.islice(origin_loader, cfg.limit_generate_batches)
            ):
                ts = torch.linspace(
                    cfg.time_padding,
                    1.0 - cfg.time_padding,
                    cfg.n_sampling_steps,
                    device=device,
                )
                x0 = batch["image"].unsqueeze(1).to(device)
                history_x, model_output = solver.solve(
                    model, interp_fn, loss_fn, x0, ts
                )

                last_x = history_x[-1]
                best_estimate = loss_fn.best_estimate(
                    last_x, model_output[-1], interp_fn, ts[-1]
                )

                generated.append(history_x[-1])
                best_estimates.append(best_estimate)

                history_x_torch = torch.stack(history_x, dim=0)
                history_dx = history_x_torch[1:] - history_x_torch[:-1]

                path_length = torch.abs(history_dx).sum(dim=[0, 2, 3, 4]).mean()
                path_lengths.append(path_length)
                n_samples.append(x0.shape[0])
                x0s.append(x0)

            path_lengths = torch.tensor(path_lengths)
            n_samples = torch.tensor(n_samples)
            path_length = (path_lengths * n_samples).sum() / n_samples.sum()
            mlflow.log_metric("PathLength", path_length.item())

            generated = torch.concat(generated, dim=0)
            best_estimates = torch.concat(best_estimates, dim=0)
            x0 = torch.concat(x0s, dim=0)

            plot_generated_grid(generated.cpu().numpy())
            plot_generated_grid(best_estimates.cpu().numpy(), label="x1")
            plot_generated_compare_grid(x0.cpu().numpy(), best_estimates.cpu().numpy())

            ## Generate descriptors for generated.
            gen_descriptors = classifier.embed(best_estimates.squeeze())
            gen_mu = gen_descriptors.mean(dim=0)
            gen_cov = (
                torch.matmul((gen_descriptors - gen_mu).T, (gen_descriptors - gen_mu))
                / gen_descriptors.shape[0]
            )

            ## Compute the Fréchet inception distance
            # See: https://minibatchai.com/2022/07/23/FID.html
            mu_dist = torch.square(ref_mu - gen_mu).sum()

            cov_mm = torch.matmul(ref_cov, gen_cov)
            eigenvals = torch.linalg.eigvals(cov_mm)
            sqrt_eigvals_sum = torch.sqrt(eigenvals.real.clamp(min=0)).sum()

            sum_traces = torch.trace(ref_cov) + torch.trace(gen_cov)

            fid = mu_dist + sum_traces - 2 * sqrt_eigvals_sum

            logger.info("FID: %s", fid.item())
            mlflow.log_metric("FID", fid.item())

            mean_distance = torch.abs(x0 - best_estimates).sum(dim=[2, 3]).mean()
            mlflow.log_metric("SampleDistance", mean_distance.item())

            gen_classifs = torch.bincount(
                classifier.classify(gen_descriptors).argmax(dim=1), minlength=10
            )
            precision = gen_classifs[run_cfg.destination_digit] / gen_classifs.sum()

            mlflow.log_metric("ClassifierPrecision", precision.item())

            ## Test variety of sample from one x0.
            random_idx = random.randint(0, len(origin_dataset) - 1)
            width, height = x0.shape[2], x0.shape[3]
            x0 = (
                origin_dataset[random_idx]["image"]
                .to(device)
                .reshape(1, 1, width, height)
            ).expand(100, -1, -1, -1)

            ts = torch.linspace(
                cfg.time_padding,
                1.0 - cfg.time_padding,
                cfg.n_sampling_steps,
                device=device,
            )
            history_x, model_output = solver.solve(model, interp_fn, loss_fn, x0, ts)
            generated_x1 = model_output[-1][:, 1]
            generated_x1[0] = x0[0]

            best_estimate = (
                loss_fn.best_estimate(
                    history_x[-1], model_output[-1], interp_fn, ts[-1]
                )
                .cpu()
                .numpy()
            )
            plot_generated_grid(best_estimate, label="x1_same_origin", shuffle=False)
    finally:
        mlflow.end_run()


def plot_generated_grid(generated: np.ndarray, label="generated", shuffle=True):
    width = 10
    height = 10
    fig, axs = plt.subplots(10, 10, figsize=(width, height))

    # Select a subsample of the generated images.
    if shuffle:
        indices = np.random.choice(generated.shape[0], width * height, replace=False)
    else:
        indices = np.arange(width * height)
    subset = generated[indices].squeeze()

    for i, img in enumerate(subset):
        ax = axs[i // width, i % width]

        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")

    fig.subplots_adjust(wspace=0, hspace=0)

    filename = f"test_{label}.png"
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()


def plot_generated_compare_grid(x0: np.ndarray, x1: np.ndarray, label="path"):
    width = 10
    height = 10
    fig, axs = plt.subplots(10, 10, figsize=(width, height))

    # Select a subsample of the generated images.
    indices = np.random.choice(x0.shape[0], width * height, replace=False)
    x0_subset = x0[indices].squeeze()
    x1_subset = x1[indices].squeeze()

    for i in range(width * height):
        ax = axs[i // width, i % width]

        img = (
            1.0
            - (
                np.stack(
                    [x0_subset[i], x1_subset[i], -1 * np.ones_like(x1_subset[i])],
                    axis=-1,
                )
                + 1.0
            )
            / 2.0
        )

        ax.imshow(img)
        ax.axis("off")

    fig.subplots_adjust(wspace=0, hspace=0)

    filename = f"test_{label}.png"
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()
