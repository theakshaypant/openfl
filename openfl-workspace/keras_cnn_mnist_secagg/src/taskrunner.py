# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten

from openfl.federated.task.runner_sa import SATaskRunner


class KerasCNN(SATaskRunner):
    """A basic convolutional neural network model."""

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        self.initialize_tensorkeys_for_functions()
