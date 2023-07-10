import copy
from typing import Optional

from torch.utils.data import ConcatDataset, Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np

from .psy_dataset import PsyDataset
from .collator import DialogueDataCollator

PSY_DATASETS = [
    "psychodiagnosis"
]


def get_dataset_name_and_kwargs_from_data_config(data_config):
    if isinstance(data_config, dict):
        name = list(data_config.keys())[0]

        # first copy the dict, then remove the size and fraction
        kwargs = copy.deepcopy(data_config[name])

        kwargs.pop("fraction", None)
        kwargs.pop("size", None)
        return name, kwargs
    else:
        return data_config, {}

def get_one_dataset(
    conf,
    dataset_name: str,
    val_split: float = 0.2,
    data_path: str = None,
    mode: str = "sft",
    max_val_set: Optional[int] = None,
    **kwargs,
) -> tuple[Dataset, Dataset | None]:

    data_path = data_path or conf.data_dir
    dataset_name = dataset_name.lower()

    if dataset_name in PSY_DATASETS:
        dataset = PsyDataset(dataset_name, data_path, "train")
        if not dataset.no_val:
            eval = PsyDataset(dataset_name, data_path, "validation")
            train = dataset
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # if eval not already defined
    if not ("eval" in locals() and "train" in locals()):
        train, eval = train_val_dataset(dataset, val_split=val_split)

    if eval and max_val_set and len(eval) > max_val_set:
        subset_indices = np.random.choice(len(eval), max_val_set)
        eval = Subset(eval, subset_indices)

    return train, eval


def train_val_dataset(dataset, val_split=0.2) -> tuple[Dataset, Dataset | None]:
    if val_split == 0:
        return dataset, None

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=666, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_dataset(conf, mode: str = "sft"):
    train_datasets, evals = [], {}

    for data_config in conf.datasets + conf.datasets_extra:
        dataset_name, kwargs = get_dataset_name_and_kwargs_from_data_config(data_config)
        train, val = get_one_dataset(conf, dataset_name, mode=mode, **kwargs)
        train_datasets.append(train)

        if val is not None:
            evals[dataset_name] = Subset(val, list(range(min(len(val), conf.eval_size)))) if conf.eval_size else val

    train = ConcatDataset(train_datasets)

    return train, evals
