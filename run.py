"""Entry point for CoLOR training runs.

Usage examples:
    python run.py dataset=cifar100 models=precision_at_recall \\
        seed=42 ood_class=2 ood_class_ratio=0.5 fraction_ood_class=0.35

    python run.py dataset=amazon_reviews datamodule=amazon_reviews_split_module \\
        models=nnPU seed=42 ood_class=0 ood_class_ratio=0.05 fraction_ood_class=0.1

    python run.py dataset=sun397 datamodule=sun397_datamodule \\
        models=BODASaito seed=42 ood_class=0 ood_class_ratio=0.5 fraction_ood_class=0.13

By default wandb runs offline. To log online, override:
    python run.py logger.offline=False logger.entity=<your-entity> logger.project=<your-project> ...
"""
import os
import time
from os.path import join

import hydra
import wandb
from omegaconf import DictConfig

from src.train import train
from src.utils import filter_config, get_dict_hash
from src.simple_utils import load_pickle, dump_pickle

os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

SUPPORTED_DATASETS = {"cifar100", "amazon_reviews", "sun397"}


def _build_run_name(config: DictConfig) -> str:
    """Compose a descriptive wandb run name from the config."""
    method = config.models._target_.split(".")[-2]
    if method == "nnPU":
        method = "nnPU" if config.nnPU else "uPU"

    shift = "no_shift_" if config.no_shift else "shift_"

    parts = [
        shift,
        method,
        f"seed_{config.seed}",
        config.arch,
        f"ns_{config.num_source_classes}",
        f"lr_{config.learning_rate}",
        f"dlr_{config.dual_learning_rate}",
        f"cp_{config.constrained_penalty}",
        f"ood_{config.ood_class}",
        f"oodr_{config.ood_class_ratio}",
        f"tprec_{config.target_precision}",
        f"clip_{config.clip}",
        f"labels_{config.use_labels}",
    ]
    return "_".join(str(p) for p in parts)


@hydra.main(config_path="config/", config_name="config.yaml")
def main(config: DictConfig):
    if config.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset={config.dataset} is not supported in this public release. "
            f"Supported: {sorted(SUPPORTED_DATASETS)}."
        )

    timestr = time.strftime("%Y%m%d-%H%M%S")

    method = config.models._target_.split(".")[-2]
    group = f"{method}_{config.dataset}_seed_{config.seed}"
    run_name = _build_run_name(config)

    wandb_kwargs = dict(
        project=config.logger.project,
        group=group,
        name=run_name,
        reinit=True,
    )
    if config.logger.entity:
        wandb_kwargs["entity"] = config.logger.entity
    if bool(config.logger.offline):
        wandb_kwargs["mode"] = "offline"

    run = wandb.init(**wandb_kwargs)

    # Group hash for hyperparameter bookkeeping (used by some methods to cache state)
    group_dict = dict(filter_config(config.datamodule), **filter_config(config.models))
    group_hash = get_dict_hash(group_dict)
    config.logger.group = group_hash
    print(group_dict)

    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir, exist_ok=True)

    hash_dict_fname = join(config.log_dir, "hash_dict.pkl")
    if os.path.isfile(hash_dict_fname):
        hash_dict = load_pickle(hash_dict_fname)
    else:
        hash_dict = dict()
    hash_dict[group_hash] = group_dict
    dump_pickle(hash_dict, hash_dict_fname)

    raw_path = join(config.log_dir, "raw")
    if not os.path.isdir(raw_path):
        os.makedirs(raw_path, exist_ok=True)

    train(config)
    run.finish()


if __name__ == "__main__":
    main()
