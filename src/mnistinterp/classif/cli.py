import hydra
import omegaconf as oc


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def cli(cfg: oc.DictConfig):
    pass
