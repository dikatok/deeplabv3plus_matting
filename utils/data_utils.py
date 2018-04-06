import tensorflow as tf
from tensorflow.contrib import image as contrib_image
from tensorflow.contrib import data


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


def _create_parse_record_fn(image_size: (int, int)):
    """Create tf record parser function

    Args:
        image_size (int, int): Desired image and mask size

    Returns:
        parser function
    """

    def parse_record_fn(example: tf.train.Example):
        """Tf record parser function (single record)

        Args:
            example (tf.train.Example): Tf record example to parse

        Returns:
            parsed image and mask
        """

        features = {
            "image": tf.FixedLenFeature((), tf.string),
            "mask": tf.FixedLenFeature((), tf.string),
            "image_filename": tf.FixedLenFeature((), tf.string),
            "mask_filename": tf.FixedLenFeature((), tf.string)
        }

        parsed_example = tf.parse_single_example(example, features)

        image = tf.cast(tf.image.decode_jpeg(parsed_example["image"], channels=3), dtype=tf.float32)
        mask = tf.cast(tf.image.decode_jpeg(parsed_example["mask"], channels=1), dtype=tf.float32) / 255.

        image = tf.image.resize_images(image, image_size)
        mask = tf.image.resize_images(mask, image_size)

        mask = tf.cast(mask, dtype=tf.int32)

        return image, mask

    return parse_record_fn


def _create_one_shot_iterator(
        tfrecord_filenames: [str],
        num_epochs: int,
        batch_size: int,
        image_size: (int, int)):
    """Create one shot iterator

    Args:
        tfrecord_filenames ([str]): List of tf record filename
        num_epochs (int): Epochs to repeat
        batch_size (int): Batch size
        image_size ((int, int)): Image and mask size

    Returns:
        one shot iterator
    """

    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    dataset = dataset.map(_create_parse_record_fn(image_size))
    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    return dataset.make_one_shot_iterator()


def _create_initializable_iterator(
        tfrecord_filenames: [str],
        num_epochs: int,
        batch_size: int,
        image_size: (int, int)):
    """Create initializable iterator

    Args:
        tfrecord_filenames ([str]): List of tf record filename
        num_epochs (int): Epochs to repeat
        batch_size (int): Batch size
        image_size ((int, int)): Image and mask size

    Returns:
        initializable iterator
    """

    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    dataset = dataset.map(_create_parse_record_fn(image_size))
    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    return dataset.make_initializable_iterator()


def _preprocess(
        images: tf.Tensor,
        masks: tf.Tensor):
    """Preprocess function

    Args:
        images (tf.Tensor): images
        masks (tf.Tensor): masks

    Returns:
        images and masks
    """

    if is_training and label is None:
        raise ValueError('During training, label must be provided.')
    if model_variant is None:
        tf.logging.warning('Default mean-subtraction is performed. Please specify '
                           'a model_variant. See feature_extractor.network_map for '
                           'supported model variants.')

    # Keep reference to original image.
    original_image = image

    processed_image = tf.cast(image, tf.float32)

    if label is not None:
        label = tf.cast(label, tf.int32)

    # Resize image and label to the desired range.
    if min_resize_value is not None or max_resize_value is not None:
        [processed_image, label] = (
            preprocess_utils.resize_to_range(
                image=processed_image,
                label=label,
                min_size=min_resize_value,
                max_size=max_resize_value,
                factor=resize_factor,
                align_corners=True))
        # The `original_image` becomes the resized image.
        original_image = tf.identity(processed_image)

    # Data augmentation by randomly scaling the inputs.
    scale = preprocess_utils.get_random_scale(
        min_scale_factor, max_scale_factor, scale_factor_step_size)
    processed_image, label = preprocess_utils.randomly_scale_image_and_label(
        processed_image, label, scale)
    processed_image.set_shape([None, None, 3])

    # Pad image and label to have dimensions >= [crop_height, crop_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    # Pad image with mean pixel value.
    mean_pixel = tf.reshape(
        feature_extractor.mean_pixel(model_variant), [1, 1, 3])
    processed_image = preprocess_utils.pad_to_bounding_box(
        processed_image, 0, 0, target_height, target_width, mean_pixel)

    label = preprocess_utils.pad_to_bounding_box(label, 0, 0, target_height, target_width, ignore_label)

    # Randomly crop the image and label.
    processed_image, label = preprocess_utils.random_crop(
        [processed_image, label], crop_height, crop_width)

    processed_image.set_shape([crop_height, crop_width, 3])

    label.set_shape([crop_height, crop_width, 1])

        # Randomly left-right flip the image and label.
    processed_image, label, _ = preprocess_utils.flip_dim(
        [processed_image, label], _PROB_OF_FLIP, dim=1)

    return processed_image, label


def create_inputs_fn(
        tfrecord_filenames: [str],
        num_epochs: int,
        batch_size: int,
        image_size: (int, int),
        scope: str,
        is_training: bool):

    """Create input function

    Args:
        tfrecord_filenames ([str]): List of tf record filename
        num_epochs (int): Epochs to repeat
        batch_size (int): Batch size
        image_size ((int, int)): Image size
        scope (str): Variable scope name

    Returns:
        input function and hook to initialize the iterator
    """

    iter_init_hook = IteratorInitializerHook()

    def inputs_fn():
        with tf.variable_scope(scope):
            iterator = _create_initializable_iterator(tfrecord_filenames, num_epochs, batch_size, image_size)

            images, masks = iterator.get_next()

            # if is_training:
            #     images, masks = _preprocess(images, masks)

            iter_init_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)

        return images, masks

    return inputs_fn, iter_init_hook
