from gan import GAN
from generator import Generator
from discriminator import Discriminator
# from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt

import gzip
import sys
import pickle



class Trainer:
    def __init__(
        self, width=28, height=28, channels=1,
        latent_size=100, epochs=50000, batch=32,
        checkpoint=50, model_type=-1
    ):
        self.width = width
        self.height = height
        self.channels = channels
        self.epochs = epochs
        self.batch = batch
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.latent_space_size = latent_size
        self.generator = Generator(
            height=self.height,
            width=self.width,
            channels=self.channels,
            latent_size=self.latent_space_size
        )
        self.discriminator = Discriminator(
            height=self.height,
            width=self.width,
            channels=self.channels
        )
        self.gan = GAN(
            generator=self.generator,
            discriminator=self.discriminator
        )
        self.load_mnist()

    def load_mnist(self, model_type: int = None):
        allowed_types = list(range(-1, 10))
        if model_type is not None:
            self.model_type = model_type

        if self.model_type not in allowed_types:
            print("ERROR: Only integers between -1 and 9 allowed")

        #mnist_files = np.load('mnist.npz')
        #self.x_train = mnist_files['x_test']
        #self.y_train = mnist_files['y_test']
        # (self.x_train, self.y_train), (_, _) = mnist.load_data()

        f = gzip.open('mnist.pkl.gz', 'rb')
        mnist_data = pickle.load(f, encoding='bytes')
        f.close()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist_data

        
        if self.model_type != -1:
            self.x_train = self.x_train[
                np.where(self.y_train == int(self.model_type))[0]
            ]
            self.x_train = (np.float32(self.x_train) - 127.5) / 127.5
            self.x_train = np.expand_dims(self.x_train, axis=3)

        return

    def train(self):
        for epoch in range(self.epochs):
            # Grab a batch.
            count_real_images = int(self.batch / 2)
            starting_index = randint(
                0, len(self.x_train) - count_real_images
            )
            real_images_raw = self.x_train[
                starting_index:
                starting_index + count_real_images
            ]
            x_real_images = real_images_raw.reshape(
                count_real_images, self.width, self.height, self.channels
            )
            y_real_labels = np.ones([count_real_images, 1])

            # Grab Generated Images for this training batch.
            latent_space_samples = self.sample_latent_space(count_real_images)
            x_generated_images = self.generator.generator.predict(latent_space_samples)
            y_generated_labels = np.zeros([self.batch - count_real_images, 1])

            # Combine to train on the discriminator
            x_batch = np.concatenate([x_real_images, x_generated_images])
            y_batch = np.concatenate([y_real_labels, y_generated_labels])

            # Now, train the discriminator.
            discriminator_loss = self.discriminator.discriminator.train_on_batch(
                x_batch, y_batch
            )[0]

            # Generate Noise
            x_latent_space_samples = self.sample_latent_space(self.batch)
            y_generated_labels = np.ones([self.batch, 1])
            generator_loss = self.gan.gan_model.train_on_batch(
                x_latent_space_samples, y_generated_labels
            )
            print(f"""
                Epoch: {epoch}
                Discriminator Loss: {discriminator_loss}
                Generator Loss: {generator_loss}
            """)
            if epoch % self.checkpoint == 0:
                self.plot_checkpoint(epoch)

        return

    def sample_latent_space(self, instances):
        return np.random.normal(
            0, 1, (instances, self.latent_space_size)
        )

    def plot_checkpoint(self, epoch):
        filename = f"data/sample_{epoch}.png"
        noise = self.sample_latent_space(16)
        images = self.generator.generator.predict(noise)
        # images = [np.reshape(image, [self.height, self.width]) for image in images]

        plt.figure(figsize=(10, 10))
        row1 = np.concatenate(images[:4, :, :, :], axis=1)
        row2 = np.concatenate(images[4:8, :, :, :], axis=1)
        row3 = np.concatenate(images[8:12, :, :, :], axis=1)
        row4 = np.concatenate(images[12:16, :, :, :], axis=1)
        total = np.concatenate([row1, row2, row3, row4], axis=0)
        plt.figure()
        plt.imshow(total, cmap="gray")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')

        _ = """

        # This did not work
        plt.figure(figsize=(10, 10))
        for ind in range(images.shape[0]):
            plt.subplot(4, 4, ind + 1)
            image = images[ind, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap="gray")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close('all')
        """
