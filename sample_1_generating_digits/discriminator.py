"""Discriminator element of the gan

Note: the comments labeled by (B) means it is different from the book because
of the version of tensorflow. However, syntax changes, particularly upper to lower
cases are just addressing PEP guidelines.
"""
# import sys
# import numpy as np
from keras.layers import Dense, Flatten  # , Reshape, Dropout, Input
from keras.src.layers.activation.leaky_relu import LeakyReLU
from tensorflow.keras.models import Sequential  # , Model
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


class Discriminator:
    """Discriminator class with attributes width, height, channels, capacity,
    shape, optimizer, discriminator."""

    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        self.width = width
        self.height = height
        self.channels = channels
        self.capacity = self.width * self.height * self.channels
        self.shape = (width, height, channels)
        self.optimizer = Adam(learning_rate=0.0002)  # (B) decays=8e-9
        self.discriminator = self.model()
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=self.optimizer,
            metrics=["Accuracy"]
        )
        self.summary()  # (B) self.discriminator.summary()

    def model(self, block_starting_size=128, num_blocks=4):
        """Starts the model attribute.

        Args:
            block_starting_size (int): Default to 128.
            num_blocks (int). Default to 4.
        """
        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense(self.capacity, input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int(self.capacity / 2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        return model

    def summary(self):
        return self.discriminator.summary()

    def save_model(self):
        plot_model(
            self.discriminator,
            to_file="data/discriminator_model.png"
        )
