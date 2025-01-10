# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Secure aggregation module."""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from openfl.interface.aggregation_functions.core import AggregationFunction
from openfl.utilities import LocalTensor


class SecureAggregation(AggregationFunction):
    """
    SecureAggregation class for performing secure aggregation of local tensors.
    """
    def call(
        self,
        local_tensors: list[LocalTensor],
        db_iterator: Iterator[pd.Series],
        tensor_name: str,
        fl_round: int,
        tags: Tuple[str],
    ) -> np.ndarray:
        """
        Perform secure aggregation by calling the appropriate aggregation
        methods based on the tags.

        Args:
            local_tensors (list[LocalTensor]): List of local tensors to be
                aggregated.
            db_iterator (Iterator[pd.Series]): Iterator over the database
                series.
            tensor_name (str): Name of the tensor.
            fl_round (int): Federated learning round number.
            tags (Tuple[str]): Tags indicating the type of aggregation to
                perform.

        Returns:
            np.ndarray: Aggregated tensor.
        """
        self._aggregate_public_keys(local_tensors, tags)
        self._aggregate_ciphertexts(local_tensors, tags)
        self._aggregate_deciphertexts(local_tensors, tags)

    def _aggregate_public_keys(
        self,
        local_tensors: list[LocalTensor],
        tags: Tuple[str],
    ):
        """
        Aggregate public keys from the local tensors.

        Args:
            local_tensors (list[LocalTensor]): List of local tensors
                containing public keys.
            tags (Tuple[str]): Tags indicating the type of aggregation to
                perform.

        Returns:
            np.ndarray: Aggregated public keys tensor.
        """
        aggregated_tensor = []
        if "public_key" in tags:
            # Setting indices for the collaborators.
            index = 1
            for tensor in local_tensors:
                # tensor[0] is public_key_1
                # tensor[1] is public_key_2
                aggregated_tensor.append(
                    [index, tensor.tensor[0], tensor.tensor[1]]
                )
                index += 1

            return np.array(aggregated_tensor)

    def _aggregate_ciphertexts(
        self,
        local_tensors: list[LocalTensor],
        tags: Tuple[str],
    ):
        """
        Aggregate ciphertexts from the local tensors.

        Args:
            local_tensors (list[LocalTensor]): List of local tensors
                containing ciphertexts.
            tags (Tuple[str]): Tags indicating the type of aggregation to
                perform.

        Returns:
            np.ndarray: Aggregated ciphertexts tensor.
        """
        aggregated_tensor = []
        if "ciphertext" in tags:
            aggregated_tensor = [tensor.tensor for tensor in local_tensors]

            return np.array(aggregated_tensor)

    def _aggregate_deciphertexts(
        self,
        local_tensors: list[LocalTensor],
        tags: Tuple[str],
    ) -> np.ndarray:
        """
        Aggregate deciphertexts from the local tensors.

        Args:
            local_tensors (list[LocalTensor]): List of local tensors
                containing deciphertexts.
            tags (Tuple[str]): Tags indicating the type of aggregation to
                perform.

        Returns:
            np.ndarray: Aggregated deciphertexts tensor.
        """
        aggregated_tensor = []
        if "deciphertext" in tags:
            aggregated_tensor = [tensor.tensor for tensor in local_tensors]

            return np.array(aggregated_tensor)
