import tensorflow as tf


def conv(inputs,
         filters: int,
         kernel_size: int,
         name: str,
         strides: int = 1,
         dilation_rate: int = 1,
         with_relu: bool = False,
         with_bn: bool = False):

    with tf.variable_scope(name):

        outputs = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding="same",
            name="conv")(inputs)

        if with_bn:
            outputs = tf.keras.layers.BatchNormalization(
                name="bn")(outputs)

        if with_relu:
            outputs = tf.keras.layers.Activation(
                "relu",
                name="relu")(outputs)

    return outputs


def global_average_pooling(inputs,
                           name: str):
    with tf.variable_scope(name):
        outputs = tf.keras.backend.mean(inputs, axis=-1, keepdims=True)

    return outputs


def resize_bilinear(inputs,
                    target_size: (int, int),
                    name: str):

    outputs = tf.keras.layers.Lambda(
        lambda x: tf.image.resize_bilinear(x, target_size),
        name=name)(inputs)

    return outputs
