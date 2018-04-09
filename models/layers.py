import tensorflow as tf


def conv(inputs,
         filters: int,
         kernel_size: int,
         name: str,
         is_training: bool,
         strides: int = 1,
         dilation_rate: int = 1,
         with_relu: bool = False,
         with_bn: bool = False):
    """Template to create convolution block

    Arguments:
        inputs
        filters
        kernel_size
        name
        is_training: whether current mode is training or not
        strides
        dilation_rate
        with_bn: whether to add batchnorm layer after conv
        with_relu: whether to add relu layer after bn (or conv if with_bn == False)
    """

    with tf.variable_scope(name):
        outputs = tf.layers.conv2d(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            use_bias=False,
            padding="same",
            name="conv")

        if with_bn:
            outputs = tf.layers.batch_normalization(
                outputs,
                axis=-1,
                training=is_training,
                name="bn")

        if with_relu:
            outputs = tf.nn.relu6(outputs, name="relu")

    return outputs


def separable_conv(inputs,
                   filters: int,
                   kernel_size: int,
                   is_training: bool,
                   name: str,
                   strides: int = 1,
                   dilation_rate: int = 1,
                   with_depth_relu: bool = False):
    """Template to create separable convolution block

    Capability to add bn and relu between depthwise and pointwise convolutions

    Arguments:
        inputs
        filters
        kernel_size
        is_training: whether current mode is training or not
        name
        strides
        dilation_rate
        with_depth_relu: whether to add relu between depthwise and pointwise convolutions

    Returns:
        separable conv output
    """

    with tf.variable_scope(name):
        separable_w = tf.get_variable(
            name="separable_weight",
            shape=[kernel_size, kernel_size, inputs.shape.as_list()[-1], 1],
            trainable=True)

        outputs = inputs

        if not with_depth_relu:
            outputs = tf.nn.relu6(outputs, name="relu_pre")

        outputs = tf.nn.depthwise_conv2d(
            outputs,
            filter=separable_w,
            strides=[1, strides, strides, 1],
            rate=[dilation_rate, dilation_rate],
            padding="SAME",
            name="conv_depthwise")

        outputs = tf.layers.batch_normalization(
            outputs,
            axis=-1,
            training=is_training,
            name="bn_depthwise")

        if with_depth_relu:
            outputs = tf.nn.relu6(outputs, name="relu_depthwise")

        outputs = tf.layers.conv2d(
            outputs,
            filters=filters,
            kernel_size=1,
            use_bias=False,
            padding="same",
            name="conv_pointwise")

        outputs = tf.layers.batch_normalization(
            outputs,
            axis=-1,
            training=is_training,
            name="bn_pointwise")

        if with_depth_relu:
            outputs = tf.nn.relu6(outputs, name="relu_post")

    return outputs


def global_average_pooling(inputs,
                           name: str):
    """Global average pooling

    Arguments:
        inputs
        name

    Returns:
        pooling output
    """

    outputs = tf.reduce_mean(
        inputs,
        axis=-1,
        keepdims=True,
        name=name)

    return outputs


def resize_bilinear(inputs,
                    target_size: (int, int),
                    name: str):
    """Resize bilinear

    Arguments:
        inputs
        target_size
        name

    Returns:
        resize output
    """

    outputs = tf.image.resize_bilinear(
        inputs,
        size=target_size,
        name=name)

    return outputs
