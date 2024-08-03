import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import numpy as np
import keras
from keras.losses import BinaryCrossentropy
from functools import reduce
from imageio.v2 import imread, imwrite
from models import steganogan_encoder_dense_model, steganogan_decoder_dense_model, steganogan_critic_model
from utils import text_to_bits, bits_to_text


class KerasSteganoGAN(keras.Model):
  def __init__(self, encoder=None, decoder=None, critic=None, image_shape=(None, None, 3), data_depth=1, model_path=None, **kwargs):
    super(KerasSteganoGAN, self).__init__(**kwargs)
    
    self.data_depth = data_depth
    self.image_shape = image_shape
    self.height, self.width, self.channels = self.image_shape
    self.message_shape = (self.height, self.width, self.data_depth) 
    self.encoder = encoder or steganogan_encoder_dense_model(self.height, self.width, self.channels, self.data_depth)
    self.decoder = decoder or steganogan_decoder_dense_model(self.height, self.width, self.channels, self.data_depth)
    self.critic  = critic or steganogan_critic_model(self.height, self.width, self.channels)

    if model_path is not None and os.path.exists(model_path):
      self.load_weights(model_path)

    self.encoder_decoder_total_loss_tracker = keras.metrics.Mean(name="encoder_decoder_total_loss")   
    self.critic_loss_tracker = keras.metrics.Mean(name="critic_loss")
    self.similarity_loss_tracker = keras.metrics.Mean(name="similarity_loss")
    self.decoding_loss_tracker = keras.metrics.Mean(name="decoding_loss")
    self.realism_loss_tracker = keras.metrics.Mean(name="realism_loss")
    self.psnr_tracker = keras.metrics.Mean(name="psnr")
    self.ssim_tracker = keras.metrics.Mean(name="ssim")
    self.bpp_tracker = keras.metrics.Mean(name="bpp")

  @property
  def metrics(self):
    return [
      self.encoder_decoder_total_loss_tracker,
      self.critic_loss_tracker,
      self.similarity_loss_tracker,
      self.decoding_loss_tracker,
      self.realism_loss_tracker,
      self.psnr_tracker,
      self.ssim_tracker,
      self.bpp_tracker
    ]
  
  def models_summary(self):
    self.critic.summary()
    self.encoder.summary()
    self.decoder.summary()

  def compile(self, encoder_optimizer, decoder_optimizer, critic_optimizer, loss_fn):
    super(KerasSteganoGAN, self).compile()
    self.encoder_optimizer = encoder_optimizer
    self.decoder_optimizer = decoder_optimizer
    self.critic_optimizer  = critic_optimizer
    self.loss_fn           = loss_fn
    
  @tf.function 
  def call(self, inputs, training=False):
    cover_image, message = inputs
    
    stego_image = self.encoder([cover_image, message], training=training)
    recovered_message = self.decoder(stego_image, training=training)

    return stego_image, recovered_message

  @tf.function
  def critic_loss(self, cover_image, stego_image):
    cover_critic_score = self.critic(cover_image, training=True)
    stego_critic_score = self.critic(stego_image, training=True)
    return cover_critic_score - stego_critic_score
  
  @tf.function
  def endoder_decoder_loss(self, cover_image, stego_image, message, recovered_message):
    similarity_loss = tf.reduce_mean(tf.square(cover_image - stego_image))
    decoding_loss = self.loss_fn(message, recovered_message) 
    realism_loss = self.critic(stego_image, training=True)

    total_loss = similarity_loss + decoding_loss + realism_loss
    
    return total_loss, similarity_loss, decoding_loss, realism_loss
  
  @tf.function
  def train_step(self, data):
    cover_image, message = data

    with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape, tf.GradientTape() as critic_tape:
      stego_image = self.encoder([cover_image, message], training=True)
      recovered_message = self.decoder(stego_image, training=True)
      
      encoder_decoder_total_loss, similarity_loss, decoding_loss, realism_loss = self.endoder_decoder_loss(cover_image, stego_image, message, recovered_message)
      critic_loss = self.critic_loss(cover_image, stego_image)

    encoder_grads = encoder_tape.gradient(encoder_decoder_total_loss, self.encoder.trainable_variables)
    decoder_grads = decoder_tape.gradient(encoder_decoder_total_loss, self.decoder.trainable_variables)
    critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

    self.encoder_optimizer.apply_gradients(zip(encoder_grads, self.encoder.trainable_variables))
    self.decoder_optimizer.apply_gradients(zip(decoder_grads, self.decoder.trainable_variables))
    self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    for p in self.critic.trainable_variables:
      p.assign(tf.clip_by_value(p, -0.1, 0.1))
            
    self.encoder_decoder_total_loss_tracker.update_state(encoder_decoder_total_loss)
    self.critic_loss_tracker.update_state(critic_loss)
    self.similarity_loss_tracker.update_state(similarity_loss)
    self.decoding_loss_tracker.update_state(decoding_loss)
    self.realism_loss_tracker.update_state(realism_loss)
    self.psnr_tracker.update_state(tf.image.psnr(cover_image, stego_image, max_val=1.0))
    self.ssim_tracker.update_state(tf.image.ssim(cover_image, stego_image, max_val=1.0))
    self.bpp_tracker.update_state(reduce(lambda x, y: x * y, message.shape[-3:]) / (cover_image.shape[1] * cover_image.shape[2]))

    return {
      'encoder_decoder_total_loss': self.encoder_decoder_total_loss_tracker.result(),
      'critic_loss': self.critic_loss_tracker.result(),
      'similarity_loss': self.similarity_loss_tracker.result(),
      'decoding_loss': self.decoding_loss_tracker.result(),
      'realism_loss': self.realism_loss_tracker.result(),
      'psnr': self.psnr_tracker.result(),
      'ssim': self.ssim_tracker.result(),
      'bpp': self.bpp_tracker.result()
    }
  
  @tf.function 
  def test_step(self, data):
    cover_image, message = data

    stego_image = self.encoder([cover_image, message], training=False)
    recovered_message = self.decoder(stego_image, training=False)

    encoder_decoder_total_loss, similarity_loss, decoding_loss, realism_loss = self.endoder_decoder_loss(cover_image, stego_image, message, recovered_message)
    critic_loss = self.critic_loss(cover_image, stego_image)

    self.encoder_decoder_total_loss_tracker.update_state(encoder_decoder_total_loss)
    self.critic_loss_tracker.update_state(critic_loss)
    self.similarity_loss_tracker.update_state(similarity_loss)
    self.decoding_loss_tracker.update_state(decoding_loss)
    self.realism_loss_tracker.update_state(realism_loss)
    self.psnr_tracker.update_state(tf.image.psnr(cover_image, stego_image, max_val=1.0))
    self.ssim_tracker.update_state(tf.image.ssim(cover_image, stego_image, max_val=1.0))
    self.bpp_tracker.update_state(reduce(lambda x, y: x * y, message.shape[-3:]) / (cover_image.shape[1] * cover_image.shape[2]))

    return {
      'encoder_decoder_total_loss': self.encoder_decoder_total_loss_tracker.result(),
      'critic_loss': self.critic_loss_tracker.result(),
      'similarity_loss': self.similarity_loss_tracker.result(),
      'decoding_loss': self.decoding_loss_tracker.result(),
      'realism_loss': self.realism_loss_tracker.result(),
      'psnr': self.psnr_tracker.result(),
      'ssim': self.ssim_tracker.result(),
      'bpp': self.bpp_tracker.result()
    }
  
  def _image_to_tensor(self, image, save_to=None, normalize='0 to 1'):
    image = tf.image.resize(image, [self.height, self.width])
    if save_to is not None:
      imwrite("images/resized_{0}".format(save_to), image.numpy().astype(np.uint8))
    image = tf.cast(image, tf.float32)
    image = tf.convert_to_tensor(image)
    if normalize == '0 to 1':
      image = image / 255.0
    elif normalize == '-1 to 1':
      image = (image / 127.5) - 1.0
    image = tf.expand_dims(image, axis=0)
    return image
  
  def _stego_tensor_to_image(self, stego_tensor):
    stego_image = tf.squeeze(stego_tensor)
    stego_image = (stego_image + 1.0) * 127.5
    stego_image = tf.cast(stego_image, tf.uint32)
    return stego_image

  def encode(self, cover_path, stego_path, message):
    cover = imread(cover_path)
    cover_tensor = self._image_to_tensor(cover, save_to=cover_path, normalize='-1 to 1')

    message = text_to_bits(message, self.message_shape)
    message = np.reshape(message, (1, self.width, self.height, self.data_depth)) 
    message = tf.convert_to_tensor(message, dtype=tf.float32)

    stego_tensor = self.encoder([cover_tensor, message])
    stego_image = self._stego_tensor_to_image(stego_tensor)
    imwrite(stego_path, stego_image.numpy().astype(np.uint8))

    ######## DEBUGGING ########
    # decoded_message_tensor = self.decoder(stego_tensor)
    # decoded_message_tensor = tf.cast(decoded_message_tensor, tf.int8)

    # print(BinaryCrossentropy(from_logits=False)(message, decoded_message_tensor))
    # print(decoded_message_tensor)
    # print(bits_to_text(decoded_message_tensor, self.message_shape))
    ######## DEBUGGING END ########

  def decode(self, stego):
    stego = imread(stego)
    stego_tensor = self._image_to_tensor(stego, save_to=None, normalize='-1 to 1')

    message_tensor = self.decoder(stego_tensor)
    message_tensor = tf.cast(message_tensor, tf.int8)

    message = bits_to_text(message_tensor, self.message_shape)
    return message