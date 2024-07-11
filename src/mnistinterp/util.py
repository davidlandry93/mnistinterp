from collections.abc import MutableMapping
import logging
import omegaconf as oc
import mlflow


logger = logging.getLogger(__name__)


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
