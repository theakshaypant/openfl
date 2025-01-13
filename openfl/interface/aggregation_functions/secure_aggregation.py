# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Secure aggregation module."""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from openfl.interface.aggregation_functions.core import AggregationFunction
from openfl.utilities import LocalTensor
from openfl.utilities.secagg import generate_agreed_key, reconstruct_secret


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
        self._rebuild_secrets(db_iterator, local_tensors, tags)

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
            np.ndarray: Aggregated public keys tensor in format
                [
                    [collaborator_id, public_key_1, public_2],
                    [collaborator_id, public_key_1, public_2],
                    ...
                ]
        """
        aggregated_tensor = []
        if "public_key" in tags:
            # Setting indices for the collaborators.
            index = 1
            for tensor in local_tensors:
                # tensor[0] is public_key_1
                # tensor[1] is public_key_2
                aggregated_tensor.append([index, tensor.tensor[0], tensor.tensor[1]])
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
            np.ndarray: Aggregated ciphertexts tensor in format
                [
                    [source_collaborator_id, destination_collaborator_id, ciphertext],
                    [source_collaborator_id, destination_collaborator_id, ciphertext],
                    ...
                ]
        """
        aggregated_tensor = []
        if "ciphertext" in tags:
            aggregated_tensor = [tensor.tensor for tensor in local_tensors]

            return np.array(aggregated_tensor)

    def _rebuild_secrets(
        self,
        db_iterator,
        local_tensors: list[LocalTensor],
        tags: Tuple[str],
    ):
        """
        Rebuild secrets from decrypted ciphertext tensors to unmask gradient
        vectors.

        This method processes a list of local tensors to reconstruct the
        secrets (seeds and private keys) required for unmasking gradient
        vectors. It creates dictionaries to store seed shares and key shares
        for each source collaborator, then reconstructs the secrets for all
        source collaborators. Finally, it generates the agreed keys for all
        collaborator permutations.

        Args:
            db_iterator: An iterator for the database.
            local_tensors (list[LocalTensor]): A list of LocalTensor objects
                containing the decrypted ciphertext tensors.
            tags (Tuple[str]): A tuple of tags indicating the type of tensors.

        Returns:
            np.ndarray: A numpy array containing the reconstructed seeds and
            the agreed keys tensor.
        """
        if "deciphertext" in tags:
            seed_shares = {}
            key_shares = {}
            for tensor in local_tensors:
                source_collaborator = tensor.tensor[0]
                dest_collaborator = tensor.tensor[1]
                if source_collaborator not in seed_shares:
                    seed_shares[source_collaborator] = {}
                if source_collaborator not in key_shares:
                    key_shares[source_collaborator] = {}
                seed_shares[source_collaborator][dest_collaborator] = tensor.tensor[2]
                key_shares[source_collaborator][dest_collaborator] = tensor.tensor[3]
            # Reconstruct the secrets (seeds and private keys) for all source
            # collaborators.
            seeds = []
            keys = {}
            for collaborator in seed_shares:
                seed = reconstruct_secret(seed_shares[collaborator])
                seeds.append([collaborator, seed])
                keys[collaborator] = reconstruct_secret(key_shares[collaborator])
            # Generate the agreed keys for all collaborator permutations.
            agreed_keys_tensor = self._generate_agreed_keys(keys, db_iterator)

            return np.array([seeds, agreed_keys_tensor])

    def _generate_agreed_keys(self, reconstructed_keys, db_iterator):
        """
        This function takes reconstructed private keys and a database iterator
        to fetch public keys, and generates agreed keys for all permutations
        of collaborators.

        Args:
            reconstructed_keys (dict): A dictionary where keys are
                collaborator IDs and values are their reconstructed private
                keys.
            db_iterator (iterable): An iterable that yields items containing
                public keys in their "tags".

        Returns:
            list: A list of lists, where each inner list contains:
                - private_key_collaborator_id (int): The ID of the
                    collaborator with the private key.
                - public_key_collaborator_id (int): The ID of the collaborator
                    with the public key.
                - agreed_key (Any): The generated agreed key for the pair of
                    collaborators.
        """
        agreed_keys = []
        # Fetch the public keys from tensor db.
        public_keys = []
        for item in db_iterator:
            if "tags" in item and "public_key" in item["tags"]:
                public_keys.append(item["nparray"])
        # Generate agreed keys for all collaborator permutations.
        for item in public_keys:
            public_key_collaborator_id = item[0]
            public_key_1 = item[1]
            for private_key_collaborator_id in reconstructed_keys:
                agreed_key = generate_agreed_key(
                    reconstructed_keys[private_key_collaborator_id], public_key_1
                )
                agreed_keys.append(
                    [private_key_collaborator_id, public_key_collaborator_id, agreed_key]
                )

        return agreed_keys
