# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
This file contains utility functions for Secure Aggregation's cipher related
operations.
"""

from typing import Tuple, Union

import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad


def create_ciphertext(
    secret_key: bytes,
    source_id: int,
    destination_id: int,
    seed_share: bytes,
    key_share: bytes,
    nonce: bytes = b"nonce",
) -> tuple[bytes, bytes, bytes]:
    """
    Creates a cipher-text using a cipher_key for collaborators source_id and
    destination_id, and share of the private seed and share of the private key.

    The function creates a byte string using the args such that
    data = b'source_id.destination_id.seed_share.key_share'.
    The "." serves as a separator such that all values used to create the
    ciphertext can be easily distinguished when decrypting.

    Args:
        secret_key (bytes): Agreed key in bytes used to construct a cipher for
            the encryption.
        source_id (int): Unique integer ID of the creating collaborator of the
            cipher text.
        destination_id (int): Unique integer ID of the recepient collaborator
            of the cipher text.
        seed_share (bytes): Share of source_id collaborator's private seed for
            destination_id collaborator.
        key_share (bytes): Share of source_id collaborator's private key for
            destination_id collaborator.

    Returns:
        bytes: Ciphertext created using the args.
        bytes: MAC tag for the ciphertext which can be used for verification.
        bytes: Nonce used for generating the cipher used for decryption.
    """
    # Converting the integer collaborator IDs to bytes.
    source_id_bytes = source_id.to_bytes(4, byteorder="big")
    destination_id_bytes = destination_id.to_bytes(4, byteorder="big")
    # Generate the byte string to be encrypted.
    data = (
        source_id_bytes + b"." + destination_id_bytes + b"." +
        seed_share + b"." + key_share
    )
    # AES cipher requires the secret key to be of a certain length.
    # We use 64 bytes as it is the maximum length available.
    padded_secret_key = pad(secret_key, 64)

    from Crypto.Random import get_random_bytes

    # Generate a random nonce to make the encryption non-deterministic.
    nonce = get_random_bytes(len(padded_secret_key) / 2)
    # Generate a ciphertext using symmetric block cipher.
    cipher = AES.new(padded_secret_key, AES.MODE_SIV, nonce=nonce)
    ciphertext, mac = cipher.encrypt_and_digest(data)

    return ciphertext, mac, nonce


def decipher_ciphertext(
    secret_key: bytes, ciphertext: bytes, mac: bytes, nonce: bytes
) -> tuple[int, int, bytes, bytes]:
    """
    Decrypt a cipher-text to get the values used to create it.

    The function uses the nonce used while creation of the ciphertext to
    create a cipher. This cipher is used to decypt the ciphertext and verify
    it using the MAC tag, which was also generated during creation of the
    ciphertext.

    Args:
        secret_key (bytes): Agreed key in bytes used to construct a cipher for
            the encryption.
        ciphertext (bytes): Ciphertext to be decrypted.
        mac (bytes): MAC tag for the ciphertext which is used for verification.
        nonce (bytes): Nonce used during cipher generation used for decryption.

    Returns:
        int: Unique integer ID of the creating collaborator of the ciphertext.
        int: Unique integer ID of the recepient collaborator of the ciphertext.
        bytes: Share of source_id collaborator's private seed for
            destination_id collaborator.
        bytes: Share of source_id collaborator's private key for
            destination_id collaborator.
    """
    # Recreate the secret key used for encryption by adding the extra padding.
    padded_secret_key = pad(secret_key, 64)
    # Generate a ciphertext using symmetric block cipher.
    cipher = AES.new(padded_secret_key, AES.MODE_SIV, nonce=nonce)

    data = cipher.decrypt_and_verify(ciphertext, mac)
    # Remove the separator "." from the decrypted data.
    # data = b'source_id.destination_id.seed_share.key_share'
    data = data.split(b".")

    return (
        # Convert the collaborator IDs to int.
        int.from_bytes(data[0], "big"),
        int.from_bytes(data[1], "big"),
        data[2],
        data[3],
    )


def pseudo_random_generator(
    seed: Union[int, float, bytes],
    shape: Tuple,
) -> np.ndarray:
    """
    Generates a random mask using a seed value passed as arg.

    Args:
        seed (Union[int, float, bytes]): Seed to be used for generating a
            pseudo-random number.
        shape (Tuple): Shape of the numpy array to be generated.

    Returns:
        np.ndarray: array with pseudo-randomly generated numbers.
    """
    if isinstance(seed, bytes):
        # If the seed is a byte string, generate a pseduo-random number using
        # it as seed and use that as seed for the numpy pseudo random
        # generator.
        import random

        random.seed(seed)
        seed = random.random()

    # Seed numpy random generator.
    rng = np.random.default_rng(seed=seed)

    return rng.random(shape)
