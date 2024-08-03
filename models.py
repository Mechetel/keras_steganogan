from keras.models import Model
from keras.layers import Input, BatchNormalization, AveragePooling2D, Conv2D, LeakyReLU
from keras.layers import Add, Concatenate, Lambda, Activation
from keras.activations import sigmoid, tanh


def steganogan_encoder_basic_model(H, W, C, D):
    """
    The BasicEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """
    Cover = Input(shape=(H, W, C), name=f'cover_image{H}x{W}x{C}')
    Message = Input(shape=(H, W, D), name=f'message_data{H}x{W}x{D}')

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b_concatenate = Concatenate(name='b_concatenate')([a, Message])
    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(b_concatenate)
    b = BatchNormalization(name='b_normalize')(b)

    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(b)
    c = BatchNormalization(name='c_normalize')(c)

    Encoder_d = Conv2D(3, kernel_size=3, padding='same', activation=tanh, name='Encoder_conv_tanh')(c)
    
    model = Model(inputs=[Cover, Message], outputs=Encoder_d, name='KerasSteganoGAN_basic_encoder')
    return model

def steganogan_encoder_residual_model(H, W, C, D):
    """
    The ResidualEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """
    Cover = Input(shape=(H, W, C), name=f'cover_image{H}x{W}x{C}')
    Message = Input(shape=(H, W, D), name=f'message_data{H}x{W}x{D}')

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b_concatenate = Concatenate(name='b_concatenate')([a, Message])
    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(b_concatenate)
    b = BatchNormalization(name='b_normalize')(b)

    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(b)
    c = BatchNormalization(name='c_normalize')(c)

    d = Conv2D(3, kernel_size=3, padding='same', name='d_conv')(c)
    
    Encoder_d = Add(name='add_C_d')([Cover, d])
    Encoder_d = Activation(tanh, name='Encoder_activation_tanh')(Encoder_d)
    
    model = Model(inputs=[Cover, Message], outputs=Encoder_d, name='KerasSteganoGAN_residual_encoder')
    return model

def steganogan_encoder_dense_model(H, W, C, D):
    """
    The DenseEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """
    Cover = Input(shape=(H, W, C), name=f'cover_image{H}x{W}x{C}')
    Message = Input(shape=(H, W, D), name=f'message_data{H}x{W}x{D}')

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b_concatenate = Concatenate(name='b_concatenate')([a, Message])
    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(b_concatenate)
    b = BatchNormalization(name='b_normalize')(b)

    c_concatenate = Concatenate(name='c_concatenate')([a, b, Message])
    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(c_concatenate)
    c = BatchNormalization(name='c_normalize')(c)

    d_concatenate = Concatenate(name='d_concatenate')([a, b, c, Message])
    d = Conv2D(3, kernel_size=3, padding='same', name='d_conv')(d_concatenate)

    Encoder_d = Add(name='add_C_d')([Cover, d])
    Encoder_d = Activation(tanh, name='Encoder_activation_tanh')(Encoder_d)
    
    model = Model(inputs=[Cover, Message], outputs=Encoder_d, name='KerasSteganoGAN_dense_encoder')
    return model

def steganogan_decoder_basic_model(H, W, C, D):
    """
    The BasicDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.
    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """
    Cover = Input(shape=(H, W, C), name=f'cover_image{H}x{W}x{C}')
    
    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(a)
    b = BatchNormalization(name='b_normalize')(b)

    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(b)
    c = BatchNormalization(name='c_normalize')(c)

    Decoder = Conv2D(D, kernel_size=3, padding='same', activation=sigmoid, name='Decoder_conv_sigmoid')(c)

    model = Model(inputs=Cover, outputs=Decoder, name='KerasSteganoGAN_basic_decoder')
    return model

def steganogan_decoder_dense_model(H, W, C, D):
    """
    The DenseDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.
    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """
    Cover = Input(shape=(H, W, C), name=f'cover_image{H}x{W}x{C}')
    
    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(a)
    b = BatchNormalization(name='b_normalize')(b)

    c_concatenate = Concatenate(name='c_concatenate')([a, b])
    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(c_concatenate)
    c = BatchNormalization(name='c_normalize')(c)

    Decoder_concatenate = Concatenate(name='Decoder_concatenate')([a, b, c])
    Decoder = Conv2D(D, kernel_size=3, padding='same', activation=sigmoid, name='Decoder_conv_sigmoid')(Decoder_concatenate)

    model = Model(inputs=Cover, outputs=Decoder, name='KerasSteganoGAN_dense_decoder')
    return model


def steganogan_critic_model(H, W, C):
    """
    The Critic module takes an image and predicts whether it is a cover
    image or a steganographic image (N, 1).
    Input: (N, 3, H, W)
    Output: (N, 1)
    """
    Stego = Input(shape=(H, W, C), name=f'stego_image{H}x{W}x{C}')

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv_1')(Stego)
    a = BatchNormalization(name='a_normalize_1')(a)

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv_2')(a)
    a = BatchNormalization(name='a_normalize_2')(a)

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv_3')(a)
    a = BatchNormalization(name='a_normalize_3')(a)

    x = Conv2D(1, kernel_size=3, padding='same', name='a_conv_4')(a)

    mean = AveragePooling2D(pool_size=(x.shape[1], x.shape[2]), name='mean')(x)
    
    model = Model(inputs=Stego, outputs=mean, name='KerasSteganoGAN_critic')
    return model