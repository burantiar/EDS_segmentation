import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout, Concatenate, Add, Activation


def conv_block(feature_map, conv_filters, do_rate):
    conv_1 = Conv2D(conv_filters, kernel_size=3, strides=1, padding='same')(feature_map)
    bn = BatchNormalization()(conv_1)
    relu = Activation(activation='relu')(bn)
    if do_rate > 0:
        do_1 = Dropout(do_rate)(relu)
        conv_2 = Conv2D(conv_filters, kernel_size=3, strides=1, padding='same')(do_1)
    else:
        conv_2 = Conv2D(conv_filters, kernel_size=3, strides=1, padding='same')(relu)

    res_conn = Conv2D(conv_filters, kernel_size=1, strides=1, padding='same')(feature_map)
    res_conn = BatchNormalization()(res_conn)
    addition = Add()([res_conn, conv_2])

    return addition


def res_block(feature_map, conv_filters, do_rate, strides=1):
    bn_1 = BatchNormalization()(feature_map)
    relu_1 = Activation(activation='relu')(bn_1)
    if do_rate > 0:
        do_1 = Dropout(do_rate)(relu_1)
        conv_1 = Conv2D(conv_filters, kernel_size=3, strides=strides, padding='same')(do_1)
    else:
        conv_1 = Conv2D(conv_filters, kernel_size=3, strides=strides, padding='same')(relu_1)
    
    bn_2 = BatchNormalization()(conv_1)
    relu_2 = Activation(activation='relu')(bn_2)
    if do_rate > 0:
        do_2 = Dropout(do_rate)(relu_2)
        conv_2 = Conv2D(conv_filters, kernel_size=3, strides=1, padding='same')(do_2)
    else:
        conv_2 = Conv2D(conv_filters, kernel_size=3, strides=1, padding='same')(relu_2)

    res_conn = Conv2D(conv_filters, kernel_size=3, strides=strides, padding='same')(feature_map)
    res_conn = BatchNormalization()(res_conn)
    addition = Add()([res_conn, conv_2])

    return addition


def ResUNet(input_shape, filters, depth, dropout_rate):

    # Input
    input = Input(shape=input_shape)

    # Encoder
    encoder_blocks = []

    enc = input

    for d in range(depth):
        f = filters * (2 ** d)
        if d == 0:
            enc = conv_block(enc, f, dropout_rate)
        else:
            enc = res_block(enc, f, dropout_rate, strides=2)

        encoder_blocks.append(enc)

    # Bottleneck
    f = filters * (2 ** depth)
    model_bottleneck = res_block(enc, f, dropout_rate, strides=2)

    # Decoder
    dec = model_bottleneck
    for d in reversed(range(depth)):
        f = filters * (2 ** d)

        up = UpSampling2D(size=2)(dec)                                                      
        conc = Concatenate()([up, encoder_blocks[d]])
        dec = res_block(conc, f, dropout_rate)

    # Output
    output = Conv2D(filters=1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(dec)
    model = tf.keras.models.Model(input, output, name='ResUNet')

    return model
