import tensorflow as tf
from models.layers import conv


NAME = "xception"


def xception_block(inputs,
                   block_filters: [int],
                   name: str,
                   final_strides: int = 1,
                   residual_type: str = None):

    residual = None
    if residual_type == "sum":
        residual = inputs
    if residual_type == "conv":
        residual = conv(
            inputs,
            filters=block_filters[-1],
            kernel_size=1,
            strides=final_strides,
            name=f"{name}_residual")

    outputs = inputs
    for i, f in enumerate(block_filters):
        strides = final_strides if i == len(block_filters) - 1 else 1
        outputs = tf.keras.layers.Activation(
            "relu",
            name=f"{name}_separable_{str(i + 1)}_relu")(outputs)
        outputs = tf.keras.layers.SeparableConv2D(
            filters=f,
            kernel_size=3,
            strides=strides,
            padding="same",
            name=f"{name}_separable_{str(i + 1)}_conv")(outputs)
        outputs = tf.keras.layers.BatchNormalization(
            name=f"{name}_separable_{str(i + 1)}_bn")(outputs)

    if residual:
        outputs = tf.keras.layers.Add()([outputs, residual])

    return outputs


# def xception_model_fn(data_format: str = "channels_first"):
#     """First class function to create xception model function
#
#     Arguments:
#         data_format (str): 'channels_first' or 'channels_last'
#
#     Returns:
#         function to create xception computation graph
#     """
#
#     def model_fn(x: tf.Tensor, is_training: bool):
#         with tf.variable_scope("xception"):
#             with tf.variable_scope("entry_flow"):
#                 x = conv(
#                     x,
#                     filters=32,
#                     kernel_size=3,
#                     strides=2,
#                     data_format=data_format,
#                     use_bn=True,
#                     use_relu=True,
#                     scope="conv1",
#                     is_training=is_training)
#                 x = conv(
#                     x,
#                     filters=64,
#                     kernel_size=3,
#                     data_format=data_format,
#                     use_bn=True,
#                     use_relu=True,
#                     scope="conv2",
#                     is_training=is_training)
#
#                 x = xception_block(
#                     x,
#                     block_filters=[128] * 3,
#                     residual_type="conv",
#                     data_format=data_format,
#                     is_training=is_training,
#                     scope="block1")
#                 x = xception_block(
#                     x,
#                     block_filters=[256] * 3,
#                     residual_type="conv",
#                     data_format=data_format,
#                     is_training=is_training,
#                     scope="block2")
#                 x = xception_block(
#                     x,
#                     block_filters=[728] * 3,
#                     final_strides=2,
#                     residual_type="conv",
#                     data_format=data_format,
#                     is_training=is_training,
#                     scope="block3")
#
#             with tf.variable_scope("middle_flow"):
#                 for i, _ in enumerate(range(8)):
#                     x = xception_block(
#                         x,
#                         block_filters=[728] * 3,
#                         residual_type="sum",
#                         data_format=data_format,
#                         is_training=is_training,
#                         scope="block" + str(i + 1))
#
#             with tf.variable_scope("exit_flow"):
#                 x = xception_block(
#                     x,
#                     block_filters=[728, 1024, 1024],
#                     final_strides=2,
#                     residual_type="conv",
#                     data_format=data_format,
#                     is_training=is_training,
#                     scope="block1")
#                 x = xception_block(
#                     x,
#                     block_filters=[1536, 1536, 2048],
#                     data_format=data_format,
#                     is_training=is_training,
#                     scope="block2")
#
#         return x
#
#     return model_fn
