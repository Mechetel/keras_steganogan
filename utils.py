import numpy as np
from collections import Counter


def pad_bits(bits, payload_shape):
  total_bits_needed = np.prod(payload_shape)
  if len(bits) < total_bits_needed:
    bits += [0] * (total_bits_needed - len(bits))
  elif len(bits) > total_bits_needed:
    raise ValueError("Message is bigger than the image")
  return bits

def text_to_bits(text, message_shape):
  """Convert text to a list of ints in {0, 1}"""
  message_shape_prod = np.prod(message_shape)

  result = []
  for c in text:
    bits = bin(ord(c))[2:]
    bits = '00000000'[len(bits):] + bits
    result.extend([int(b) for b in bits])

  message = result + [0] * 32
  payload = message
  while len(payload) < message_shape_prod:
    payload += message

  payload = payload[:message_shape_prod]
  return payload

def bits_to_text(bits, message_shape):
  """Convert a list of ints in {0, 1} to text"""
  bits = np.reshape(bits, np.prod(message_shape))
  chars = []
  for b in range(int(len(bits)/8)):
    byte = bits[b*8:(b+1)*8]
    chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))

  full_message = ''.join(chars)

  candidates = Counter()
  for candidate in full_message.split('\x00\x00\x00\x00'):
    if candidate:
      candidates[candidate] += 1

  # choose most common message
  if len(candidates) == 0:
    raise ValueError('Failed to find message.')

  candidate, count = candidates.most_common(1)[0]
  print(f'Found {count} candidates for message, choosing most common.')

  return candidate

# [0000000]
# [0000000]
# [1000000]

# =

# [0000000]