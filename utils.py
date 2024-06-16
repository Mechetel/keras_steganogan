import numpy as np
from functools import reduce


def pad_bits(bits, payload_shape):
  total_bits_needed = np.prod(payload_shape)
  if len(bits) < total_bits_needed:
    bits += [0] * (total_bits_needed - len(bits))
  elif len(bits) > total_bits_needed:
    raise ValueError("Message is bigger than the image")
  return bits

def text_to_bits(text, message_shape):
  """Convert text to a list of ints in {0, 1}"""
  result = []
  for c in text:
    bits = bin(ord(c))[2:]
    bits = '00000000'[len(bits):] + bits
    result.extend([int(b) for b in bits])
  
  result = pad_bits(result, message_shape)
  return result

def bits_to_text(bits, message_shape):
  """Convert a list of ints in {0, 1} to text"""
  bits = np.reshape(bits, reduce(lambda x, y: x * y, message_shape))
  chars = []
  for b in range(int(len(bits)/8)):
    byte = bits[b*8:(b+1)*8]
    chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
  return ''.join(chars)