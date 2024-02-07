"""
Contains the GAN class, which puts together the pieces:
the generator and discriminator.
"""
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


class GAN:
    """
    Class that puts together the GAN.
    """
    def __init__(self, discriminator, generator):
        """Initialzes the GAN by passing the discriminator and generator.

        Args:
            discriminator (Discriminator)
            generator (Generator)
        """

        self.optimizer = Adam(learning_rate=0.0002)
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator.trainable = False
        self.gan_model = self.model()
        self.gan_model.compile(
            loss="binary_crossentropy", optimizer=self.optimizer
        )
        self.gan_model.summary()

    def model(self):
        """Initializes the model."""
        model = Sequential()
        model.add(self.generator.generator)
        model.add(self.discriminator.discriminator)
        return model

    def summary(self):
        """Summarizes the model."""
        return self.gan_model.summary()

    def save_model(self):
        """Save the model plot to a file in the folder `data`"""
        plot_model(self.gan_model, to_file="data/GAN_model.png")
