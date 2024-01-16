# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy
import torch

from megatron_lm.megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron_lm.megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron_lm.megatron.core.datasets.utils import log_single_rank, normalize

logger = logging.getLogger(__name__)

_VERBOSE = False


class BlendedDataset(torch.utils.data.Dataset):
    """Conjugating class for a set of MegatronDataset instances

    Args:
        datasets (List[MegatronDataset]): The MegatronDataset instances to blend

        weights (List[float]): The weights which determines the dataset blend ratios

        size (int): The number of samples to draw from the blend

        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation

    Raises:
        RuntimeError: When the dataset has fewer or more samples than 'size' post-initialization
    """

    def __init__(
        self,
        datasets: List[MegatronDataset],
        weights: List[float],
        size: int,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        assert len(datasets) < 32767
        assert len(datasets) == len(weights)
        assert numpy.isclose(sum(weights), 1.0)
        assert all(map(lambda _: type(_) == type(datasets[0]), datasets))  # type: ignore # noqa:

        # Alert user to unnecessary blending
        if len(datasets) == 1:
            log_single_rank(
                logger, logging.WARNING, "Building a BlendedDataset for a single MegatronDataset"
            )

        # Redundant normalization for bitwise identical comparison with Megatron-LM
        weights = normalize(weights)

        self.datasets = datasets
        self.weights = weights
        self.size = size
        self.config = config

        unique_identifiers = OrderedDict()
        unique_identifiers["class"] = type(self).__name__
        unique_identifiers["datasets"] = [dataset.unique_identifiers for dataset in self.datasets]
        unique_identifiers["weights"] = self.weights
        unique_identifiers["size"] = self.size

        self.unique_description = json.dumps(unique_identifiers, indent=4)
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()

        self.dataset_index, self.dataset_sample_index = self._build_indices()

        # Check size
        _ = self[self.size - 1]
        try:
            _ = self[self.size]
            raise RuntimeError(f"{type(self).__name__} size is improperly bounded")
        except IndexError:
            log_single_rank(logger, logging.INFO, f"> {type(self).__name__} length: {len(self)}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        dataset_id = self.dataset_index[idx]
        dataset_sample_id = self.dataset_sample_index[idx]
        return {
            "dataset_id": dataset_id,
            **self.datasets[dataset_id][dataset_sample_id],  # type: ignore
        }

    def _build_indices(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Build and optionally cache the dataset index and the dataset sample index

        The dataset index is a 1-D mapping which determines the dataset to query. The dataset
        sample index is a 1-D mapping which determines the sample to request from the queried
        dataset.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The dataset index and the dataset sample index
        """
        path_to_cache = self.config.path_to_cache

        if path_to_cache:
            get_path_to = lambda suffix: os.path.join(  # noqa
                path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}"
            )
            path_to_description = get_path_to("description.txt")
            path_to_dataset_index = get_path_to("dataset_index.npy")
            path_to_dataset_sample_index = get_path_to("dataset_sample_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [path_to_description, path_to_dataset_index, path_to_dataset_sample_index],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (not cache_hit and torch.distributed.get_rank() == 0):
            log_single_rank(
                logger, logging.INFO, f"Build and save the {type(self).__name__} indices",
            )

            # Build the dataset and dataset sample indexes
            log_single_rank(
                logger, logging.INFO, "\tBuild and save the dataset and dataset sample indexes"
            )
            t_beg = time.time()
            from megatron_lm.megatron.core.datasets import helpers

            dataset_index = numpy.zeros(self.size, dtype=numpy.int16)
            dataset_sample_index = numpy.zeros(self.size, dtype=numpy.int64)
            helpers.build_blending_indices(
                dataset_index,
                dataset_sample_index,
                self.weights,
                len(self.datasets),
                self.size,
                _VERBOSE,
            )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:  # type: ignore
                    writer.write(self.unique_description)
                # Save the indexes
                numpy.save(path_to_dataset_index, dataset_index, allow_pickle=True)  # type: ignore
                numpy.save(path_to_dataset_sample_index, dataset_sample_index, allow_pickle=True)  # type: ignore
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "Unable to save the indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            return dataset_index, dataset_sample_index

        log_single_rank(logger, logging.INFO, f"Load the {type(self).__name__} indices")

        log_single_rank(
            logger, logging.INFO, f"\tLoad the dataset index from {path_to_dataset_index}"  # type: ignore
        )
        t_beg = time.time()
        dataset_index = numpy.load(path_to_dataset_index, allow_pickle=True, mmap_mode='r')  # type: ignore
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the dataset sample index from {path_to_dataset_sample_index}",  # type: ignore
        )
        t_beg = time.time()
        dataset_sample_index = numpy.load(
            path_to_dataset_sample_index, allow_pickle=True, mmap_mode='r'  # type: ignore
        )
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        return dataset_index, dataset_sample_index
