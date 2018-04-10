import tensorflow as tf

from models.backbones import xception, mobilenetv2
from models.layers import conv, separable_conv, global_average_pooling, resize_bilinear


def _image_pooling(inputs,
                   filters: int,
                   target_size: (int, int),
                   is_training: bool):
    """Pooling block

    Consists of global_avg_pool_2d -> conv -> resize

    Arguments:
        inputs
        filters
        target_size: pooling output
        is_training: whether current mode is training or not

    Returns:
        pooling output
    """

    with tf.variable_scope("image_pooling"):
        outputs = global_average_pooling(
            inputs,
            name="pool")

        outputs = conv(
            outputs,
            filters=filters,
            kernel_size=1,
            with_bn=True,
            with_relu=True,
            is_training=is_training,
            name="conv")

        outputs = resize_bilinear(
            outputs,
            target_size=target_size,
            name="resize")

    return outputs


def _aspp(inputs,
          filters: int,
          atrous_rates: [int],
          is_training: bool):
    """ASPP block

    Consists of concat(conv1x1, 3x conv3x3, pool) -> conv -> dropout

    Arguments:
        inputs
        filters
        atrous_rates: atrous rates to be used, must be list of 3 integers
        is_training: whether current mode is training or not

    Returns:
        ASPP output
    """

    assert len(atrous_rates) == 3

    with tf.variable_scope("atrous_spatial_pyramid_pooling"):
        conv1x1 = conv(
            inputs,
            filters=filters,
            kernel_size=1,
            with_relu=True,
            with_bn=True,
            is_training=is_training,
            name="conv1x1")

        conv3x3_1 = separable_conv(
            inputs,
            filters=filters,
            kernel_size=3,
            dilation_rate=atrous_rates[0],
            with_depth_relu=True,
            is_training=is_training,
            name="conv3x3r6")

        conv3x3_2 = separable_conv(
            inputs,
            filters=filters,
            kernel_size=3,
            dilation_rate=atrous_rates[1],
            with_depth_relu=True,
            is_training=is_training,
            name="conv3x3r12")

        conv3x3_3 = separable_conv(
            inputs,
            filters=filters,
            kernel_size=3,
            dilation_rate=atrous_rates[2],
            with_depth_relu=True,
            is_training=is_training,
            name="conv3x3r18")

        pool_size = inputs.shape.as_list()[1:3]

        pool = _image_pooling(
            inputs,
            filters=filters,
            target_size=pool_size,
            is_training=is_training)

        outputs = tf.concat(
            [conv1x1,
             conv3x3_1,
             conv3x3_2,
             conv3x3_3,
             pool],
            axis=-1)

        outputs = conv(
            outputs,
            filters=filters,
            kernel_size=1,
            name="projection",
            with_relu=True,
            with_bn=True,
            is_training=is_training)

        outputs = tf.layers.dropout(
            outputs,
            rate=0.5,
            name="dropout")

    return outputs


def _encoder(inputs,
             backbone_model_fn,
             output_stride: int,
             is_training: bool):
    """Deeplabv3 encoder block

    Consists of features_extraction -> ASPP

    Arguments:
        inputs
        backbone_model_fn: function to create backbone features extractor
        output_stride: define dilation rate and strides rate to use in encoder, either 8 or 16
        is_training: whether current mode is training or not

    Returns:
        encoder output
    """

    assert output_stride in [8, 16]  # output stride can only be either 8 or 16

    with tf.variable_scope("encoder"):
        backbone_out = backbone_model_fn(inputs, is_training=is_training)  # features extractor output

        if output_stride == 8:
            atrous_rates = (12, 24, 36)
        else:
            atrous_rates = (6, 12, 18)

        aspp_out = _aspp(
            backbone_out,
            filters=256,
            atrous_rates=atrous_rates,
            is_training=is_training)  # atrous spatial pyramid pooling

    return aspp_out


def _decoder(low_level_features,
             aspp_out,
             output_size: (int, int),
             is_training: bool):
    """Deeplabv3 decoder block

    Consists of concat(conv(llf), aspp) -> sepconv -> sepconv -> conv -> resize

    Arguments:
        low_level_features: features from lower layer in feature extractor
        aspp_out: encoder output
        output_size: model output size, should be equal to inputs size
        is_training: whether current mode is training or not

    Returns:
        decoder output
    """

    with tf.variable_scope("decoder"):
        low_level_features = conv(
            low_level_features,
            filters=48,
            kernel_size=1,
            name="low_level_features_projection",
            with_relu=True,
            with_bn=True,
            is_training=is_training)  # convolution on low level features

        aspp_out = resize_bilinear(
            aspp_out,
            target_size=low_level_features.shape.as_list()[1:3],
            name="aspp_resize")  # resize aspp output to low level features spatial size

        outputs = tf.concat(
            [low_level_features,
             aspp_out],
            axis=-1)  # concatentate aspp and low level features

        outputs = xception.xception_block(
            outputs,
            block_filters=[256, 256],
            final_strides=1,
            with_depth_relu=True,
            is_training=is_training,
            name="decoder_block")

        outputs = conv(
            outputs,
            filters=2,
            kernel_size=1,
            is_training=is_training,
            name="conv")

        outputs = resize_bilinear(
            outputs,
            target_size=output_size,
            name="conv_resize")

        outputs = conv(
            outputs,
            filters=2,
            kernel_size=1,
            is_training=is_training,
            name="logits_conv")  # logits

    return outputs


def create_model_fn(
        backbone: str = "xception",
        output_stride: int = 8,
        img_size: (int, int) = (299, 299)):
    """First order function which returns model_fn

    Arguments:
        backbone: whether to use 'xception' or 'mobilenet' (TODO) as features extractor
        output_stride: define dilation rate and strides rate to use in encoder, either 8 or 16
        img_size: inputs image size

    Returns:
        model_fn with signatures (features, is_training) and returns logits tensor
    """

    def model_fn(features, is_training: bool):
        if backbone == xception.NAME:
            backbone_model_fn = xception.xception_model_fn(output_stride=output_stride)
            low_level_features_name = "deeplabv3plus/encoder/xception/entry_flow/block2_separable_2/" \
                                      "bn_pointwise/FusedBatchNorm:0"
        else:
            raise ValueError("Select backbone is not supported")

        with tf.variable_scope("deeplabv3plus"):
            features = ((features / 255.) - 0.5) * 2

            aspp_out = _encoder(
                features,
                backbone_model_fn=backbone_model_fn,
                output_stride=output_stride,
                is_training=is_training)

            low_level_features = tf.get_default_graph().get_tensor_by_name(low_level_features_name)

            logits = _decoder(
                low_level_features,
                aspp_out,
                output_size=img_size,
                is_training=is_training)
        return logits

    return model_fn
