import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from imageio.v2 import imread, imwrite

import tensorflow as tf
from tensorflow import keras

from utils import text_to_bits


class SaveImages(keras.callbacks.Callback):
    def __init__(self, data_depth, image_shape, images_path, save_to, **kwargs):
        super(keras.callbacks.Callback, self).__init__(**kwargs)
        self.data_depth = data_depth
        self.image_shape = image_shape
        self.height, self.width, self.channels = self.image_shape
        self.message_shape = (self.height, self.width, self.data_depth)
        self.margin = 16
        self.num_rows = 2
        self.num_cols = 4
        self.images_path = images_path
        self.save_to = save_to

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1

        image_array = np.full((
            self.margin + (self.num_rows * (128 + self.margin)),
            self.margin + (self.num_cols * (128 + self.margin)), 3),
            255, dtype=np.uint8)

        stego_images = []
        for i in range(1, 9): # 1..8
            cover_path = f"{self.images_path}/image{i}.png"
            message = "This is a very secret message!"
            stego_image = self._encode(cover_path, message)
            stego_images.append(stego_image)

            if not os.path.exists(f"{self.save_to}/epoch_{epoch}_images"):
                os.makedirs(f"{self.save_to}/epoch_{epoch}_images")

            imwrite(f"{self.save_to}/epoch_{epoch}_images/stego_image{i}.png", stego_image.numpy().astype(np.uint8))

        for i in range(8):
            row = i // self.num_cols
            col = i % self.num_cols
            image_array[self.margin + (row * (128 + self.margin)):self.margin + (row * (128 + self.margin)) + 128,
                        self.margin + (col * (128 + self.margin)):self.margin + (col * (128 + self.margin)) + 128] = stego_images[i]

        imwrite(f"{self.save_to}/{epoch}_epoch_stego_images.png", image_array)

    def _encode(self, cover_path, message):
        cover = imread(cover_path)
        cover_tensor = tf.image.resize(cover, [self.height, self.width])
        cover_tensor = tf.cast(cover_tensor, tf.float32)
        cover_tensor = tf.convert_to_tensor(cover_tensor)
        cover_tensor = (cover_tensor / 127.5) - 1.0
        cover_tensor = tf.expand_dims(cover_tensor, axis=0)

        message = text_to_bits(message, self.message_shape)
        message = np.reshape(message, (1, self.width, self.height, self.data_depth)) 
        message = tf.convert_to_tensor(message, dtype=tf.float32)

        stego_tensor = self.model.encoder([cover_tensor, message])
        stego_image = tf.squeeze(stego_tensor)
        stego_image = (stego_image + 1.0) * 127.5
        stego_image = tf.cast(stego_image, tf.uint32)

        return stego_image

class Checkpoint(keras.callbacks.Callback):
    def __init__(self, model_path, **kwargs):
        super(keras.callbacks.Callback, self).__init__(**kwargs)
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path)