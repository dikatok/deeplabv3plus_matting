import tensorflow as tf

from models.layers import conv, separable_conv


NAME = "xception"


def xception_block(inputs,
                   block_filters: [int],
                   is_training: bool,
                   name: str,
                   final_strides: int = 1,
                   dilation_rate: int = 1,
                   with_depth_relu: bool = False,
                   residual_type: str = None):
    """Xception block template

    Arguments:
        inputs
        block_filters: number of separable conv in the block
        is_training: whether current mode is training or not
        name
        final_strides: output spatial strides
        dilation_rate
        with_depth_relu
        residual_type: either 'conv' or 'sum' or None
    """

    residual = None
    if residual_type == "sum":
        residual = inputs
    if residual_type == "conv":
        residual = conv(
            inputs,
            filters=block_filters[-1],
            kernel_size=1,
            strides=final_strides,
            is_training=is_training,
            name=f"{name}_residual")

    outputs = inputs
    for i, f in enumerate(block_filters):
        strides = final_strides if i == len(block_filters) - 1 else 1
        outputs = separable_conv(
            outputs,
            filters=f,
            kernel_size=3,
            strides=strides,
            dilation_rate=dilation_rate,
            with_depth_relu=with_depth_relu,
            is_training=is_training,
            name=f"{name}_separable_{str(i + 1)}")

    if residual is not None:
        outputs = tf.add(outputs, residual)

    return outputs


def xception_model_fn(output_stride: int):
    """First order function which returns model_fn

    Arguments:
        output_stride (int): 'channels_first' or 'channels_last'

    Returns:
        model_fn with signature (x, is_training) and returns model output
    """

    assert output_stride in [8, 16]

    if output_stride == 8:
        entry_block_stride = 1
        middle_block_rate = 2
        exit_block_rates = (2, 4)
    else:
        entry_block_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)

    def model_fn(x, is_training: bool):
        with tf.variable_scope("xception"):
            with tf.variable_scope("entry_flow"):
                x = conv(
                    x,
                    filters=32,
                    kernel_size=3,
                    strides=2,
                    with_bn=True,
                    with_relu=True,
                    is_training=is_training,
                    name="conv1")
                x = conv(
                    x,
                    filters=64,
                    kernel_size=3,
                    with_bn=True,
                    with_relu=True,
                    is_training=is_training,
                    name="conv2")

                x = xception_block(
                    x,
                    block_filters=[128] * 3,
                    final_strides=2,
                    residual_type="conv",
                    is_training=is_training,
                    name="block1")
                x = xception_block(
                    x,
                    block_filters=[256] * 3,
                    final_strides=2,
                    residual_type="conv",
                    is_training=is_training,
                    name="block2")
                x = xception_block(
                    x,
                    block_filters=[728] * 3,
                    final_strides=entry_block_stride,
                    residual_type="conv",
                    is_training=is_training,
                    name="block3")

            with tf.variable_scope("middle_flow"):
                for i, _ in enumerate(range(16)):
                    x = xception_block(
                        x,
                        block_filters=[728] * 3,
                        dilation_rate=middle_block_rate,
                        residual_type="sum",
                        is_training=is_training,
                        name="block" + str(i + 1))

            with tf.variable_scope("exit_flow"):
                x = xception_block(
                    x,
                    block_filters=[728, 1024, 1024],
                    dilation_rate=exit_block_rates[0],
                    residual_type="conv",
                    is_training=is_training,
                    name="block1")
                x = xception_block(
                    x,
                    block_filters=[1536, 1536, 2048],
                    dilation_rate=exit_block_rates[1],
                    with_depth_relu=True,
                    is_training=is_training,
                    name="block2")

        return x

    return model_fn
