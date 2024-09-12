import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


class checkpoint_callback(keras.callbacks.Callback):
    """
    Subclass of keras.callbacks.Callback to save the weights every epoch in a .keras file
    """
    def __init__(self, model_path, **kwargs):
        super(keras.callbacks.Callback, self).__init__(**kwargs)
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path)