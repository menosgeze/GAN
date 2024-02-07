from keras.layers import BatchNormalization, Dense, Reshape
from keras.src.layers.activation.leaky_relu import LeakyReLU
from tensorflow.keras.models import Sequential
import numpy as np
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


class Generator():
    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        self.width = width
        self.height = height
        self.channels = channels
        self.OPTIMIZER = Adam(learning_rate=0.0002)
        self.latent_space_size = latent_size
        self.latent_space = np.random.normal(0, 1, self.latent_space_size)
        self.generator = self.model()
        self.generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.summary()

    def model(self, block_starting_size=128, num_blocks=4):
        model = Sequential()
        block_size = block_starting_size
        model.add(
            Dense(block_size, input_shape=(self.latent_space_size,))
        )
        model.add(LeakyReLU(alpha=0.2))
        for iteration in range(num_blocks - 1):
            block_size = block_size * 2
            model.add(Dense(block_size))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))

        model.add(
            Dense(self.width * self.height * self.channels, activation='tanh')
        )
        model.add(Reshape((self.width, self.height, self.channels)))
        return model

    def summary(self):
        return self.generator.summary()

    def save_model(self):
        plot_model(
            self.generator,
            to_file='data/generator_model.png'
        )
