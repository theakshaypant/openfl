# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
This file contains utility functions for Secure Aggregation for key operations.
"""

from Crypto.PublicKey import ECC


def generate_key_pair(curve: str = "ed25519") -> tuple[ECC.EccKey, bytes]:
    """
    Generates a public-private key pair for a specific curve.

    Args:
        curve (str, optional): The curve to use for generating the key pair.
            Defaults to 'ed25519'

    Returns:
        ECC.EccKey: Private key in pycryptodome format.
        bytes: Public key as bytes.
    """
    # Generate private key.
    key = ECC.generate(curve=curve)
    # Generate public_key
    public_key = key.public_key().export_key(format='PEM')

    return key, public_key


def generate_agreed_key(
    public_key: bytes,
    private_key: ECC.EccKey,
    key_count: int = 1,
    key_length: int = 32,
) -> bytes:
    """
    Uses Diffie-Helman key agreement to generate an agreed key between a pair
    of public-private keys.

    Args:
        public_key (bytes): Public key to be used for key agreement.
        private_key (ECC.EccKey): Private key in pycryptodome format to be
            used for key agreement.
        key_count (int, optional): Number of agreed keys to be generated.
            Defaults to 1.
        key_length (int, optional): Size of each key in bytes to be generated.
            Defaults to 32.
        salt (bytes, optional): A non-secret, reusable value that strengthens
            the randomness.
            Defaults to b'nonce'.

    Returns:
        bytes: Agreed key between the two keys shared in args.
    """
    import functools
    import random

    from Crypto.Hash import SHA256
    from Crypto.Protocol.KDF import HKDF
    from Crypto.Random import get_random_bytes

    # Key derivation function.
    kdf = functools.partial(
        HKDF,
        key_len=key_length,
        # Ideally, salt should be as long as the digest size of the chosen
        # hash; SHA256 has 32 byte (256 bits) digest size.
        salt=get_random_bytes(random.randint(32)),
        hashmod=SHA256,
        num_keys=key_count,
    )

    from Crypto.Protocol.DH import key_agreement

    # Using Diffie-Hellman key agreement.
    key = key_agreement(
        static_priv=private_key,
        static_pub=ECC.import_key(public_key),
        kdf=kdf
    )
    return key
