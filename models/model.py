import tensorflow as tf

from models.backbones import xception, mobilenetv2
from models.layers import conv, global_average_pooling, resize_bilinear


def _image_pooling(inputs,
                   filters: int,
                   target_size: (int, int)):

    with tf.variable_scope("image_pooling"):
        outputs = global_average_pooling(
            inputs,
            name="pool")

        outputs = conv(
            outputs,
            filters=filters,
            kernel_size=1,
            name="conv")

        outputs = resize_bilinear(
            outputs,
            target_size=target_size,
            name="resize")

    return outputs


def _aspp(inputs,
          filters: int):

    with tf.variable_scope("atrous_spatial_pyramid_pooling"):
        conv1x1 = conv(
            inputs,
            filters=filters,
            kernel_size=1,
            name="conv1x1",
            with_relu=True,
            with_bn=True)

        conv3x3r6 = conv(
            inputs,
            filters=filters,
            kernel_size=3,
            dilation_rate=6,
            name="conv3x3r6",
            with_relu=True,
            with_bn=True)

        conv3x3r12 = conv(
            inputs,
            filters=filters,
            kernel_size=3,
            dilation_rate=12,
            name="conv3x3r12",
            with_relu=True,
            with_bn=True)

        conv3x3r18 = conv(
            inputs,
            filters=filters,
            kernel_size=3,
            dilation_rate=18,
            name="conv3x3r18",
            with_relu=True,
            with_bn=True)

        pool_size = inputs.shape.as_list()[1:3]

        pool = _image_pooling(
            inputs,
            filters=filters,
            target_size=pool_size)

        outputs = tf.keras.layers.Concatenate()([
            conv1x1,
            conv3x3r6,
            conv3x3r12,
            conv3x3r18,
            pool])

        outputs = conv(
            outputs,
            filters=filters,
            kernel_size=1,
            name="projection",
            with_relu=True,
            with_bn=True)

    return outputs


def _encoder(inputs,
             backbone_model_fn):
    inputs_shape = inputs.shape.as_list()[1:4]

    with tf.variable_scope("encoder"):
        backbone_out = backbone_model_fn(
            include_top=False,
            weights=None,
            input_shape=inputs_shape,
            input_tensor=inputs)(inputs)

        aspp_out = _aspp(backbone_out, filters=256)

    return aspp_out


def _decoder(low_level_features,
             aspp_out,
             output_size: (int, int)):

    with tf.variable_scope("decoder"):

        low_level_features = conv(
            low_level_features,
            filters=48,
            kernel_size=1,
            name="low_level_features_projection",
            with_relu=True,
            with_bn=True)

        aspp_out = resize_bilinear(
            aspp_out,
            target_size=low_level_features.shape[1:3],
            name="aspp_resize")

        outputs = tf.keras.layers.Concatenate()([
            low_level_features,
            aspp_out])

        outputs = xception.xception_block(
            outputs,
            block_filters=[256, 256],
            final_strides=1,
            name="block1")

        outputs = conv(
            outputs,
            filters=2,
            kernel_size=1,
            name="conv1")

        outputs = resize_bilinear(
            outputs,
            target_size=output_size,
            name="conv1_resize")

        outputs = conv(
            outputs,
            filters=2,
            kernel_size=1,
            name="conv2")

    return outputs


def create_model_fn(
        backbone: str = "xception",
        img_size: (int, int) = (299, 299)):
    """First class function to create model function

    Args:
        backbone (str): Backbone used for feature extraction
        img_size ((int, int)): Input image size during training
        data_format (str): Determine channels axis position

    Returns:
        function to create model graph with signature as follows:
            Args: (features, is_training)
            Returns: logits tensor
    """

    def model_fn(features):
        if backbone == xception.NAME:
            backbone_model_fn = tf.keras.applications.Xception
            low_level_features_name = "deeplabv3plus/encoder/xception/block2_sepconv2_bn/FusedBatchNorm:0"
        else:
            raise ValueError("Select backbone is not supported")

        with tf.variable_scope("deeplabv3plus"):
            aspp_out = _encoder(features, backbone_model_fn)

            low_level_features = tf.get_default_graph().get_tensor_by_name(low_level_features_name)

            logits = _decoder(
                low_level_features,
                aspp_out,
                output_size=img_size)

        return logits

    return model_fn
