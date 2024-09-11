import tensorflow as tf


def normalize_img(img):
  return (img / 127.5) - 1

def create_message_tensor_for_training(batch_size, width, height, data_depth):
  message = tf.random.uniform([batch_size, width, height, data_depth], 0, 2, dtype=tf.int32)
  message = tf.cast(message, tf.float32)
  return message

def create_message_dataset(batch_size, num_batches, width, height, data_depth):
  message_tensors = [create_message_tensor_for_training(batch_size, width, height, data_depth) for _ in range(num_batches)]
  return tf.data.Dataset.from_tensor_slices(tf.concat(message_tensors, axis=0)).batch(batch_size)