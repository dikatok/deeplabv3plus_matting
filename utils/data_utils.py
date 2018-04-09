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
    """

    :param image_size:
    :return:
    """

    def parse_record_fn(example: tf.train.Example):

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
    """

    :param tfrecord_filenames:
    :param num_epochs:
    :param batch_size:
    :param image_size:
    :return:
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
    """

    :param tfrecord_filenames:
    :param num_epochs:
    :param batch_size:
    :param image_size:
    :return:
    """

    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    dataset = dataset.map(_create_parse_record_fn(image_size))
    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    return dataset.make_initializable_iterator()


def random_crop_and_pad_image_and_labels(image, labels, size):
  """Randomly crops `image` together with `labels`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
  Returns:
    A tuple of (cropped_image, cropped_label).
  """
  combined = tf.concat([image, labels], axis=3)
  image_shape = tf.shape(image)

  last_label_dim = tf.shape(labels)[-1]
  last_image_dim = tf.shape(image)[-1]
  combined_crop = tf.random_crop(
      combined,
      size=[image_shape[0], size[0], size[1], 4])
  return (combined_crop[:, : , :, :last_image_dim],
          combined_crop[:, :, :, last_image_dim:])


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
  """Gets a random scale value.

  Args:
    min_scale_factor: Minimum scale value.
    max_scale_factor: Maximum scale value.
    step_size: The step size from minimum to maximum value.

  Returns:
    A random scale value selected between minimum and maximum value.

  Raises:
    ValueError: min_scale_factor has unexpected value.
  """
  if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
    raise ValueError('Unexpected value of min_scale_factor.')

  if min_scale_factor == max_scale_factor:
    return tf.to_float(min_scale_factor)

  # When step_size = 0, we sample the value uniformly from [min, max).
  if step_size == 0:
    return tf.random_uniform([1],
                             minval=min_scale_factor,
                             maxval=max_scale_factor)

  # When step_size != 0, we randomly select one discrete value from [min, max].
  num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
  scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
  shuffled_scale_factors = tf.random_shuffle(scale_factors)
  return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, label=None, scale=1.0):
  """Randomly scales image and label.

  Args:
    image: Image with shape [height, width, 3].
    label: Label with shape [height, width, 1].
    scale: The value to scale image and label.

  Returns:
    Scaled image and label.
  """
  # No random scaling if scale == 1.
  if scale == 1.0:
    return image, label
  image_shape = tf.shape(image)
  new_dim = tf.to_int32(tf.to_float([image_shape[1], image_shape[2]]) * scale)

  # Need squeeze and expand_dims because image interpolation takes
  # 4D tensors as input.
  image = tf.image.resize_bilinear(
      image,
      new_dim,
      align_corners=True)
  if label is not None:
    label = tf.image.resize_nearest_neighbor(
        label,
        new_dim,
        align_corners=True)
  return image, label


def _augment(
        image: tf.Tensor,
        label: tf.Tensor,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0.):

    processed_image = tf.cast(image, tf.float32)

    if label is not None:
        label = tf.cast(label, tf.float32)


    # Data augmentation by randomly scaling the inputs.
    # print(processed_image.shape)

    scale = get_random_scale(
        min_scale_factor, max_scale_factor, scale_factor_step_size)
    processed_image, label = randomly_scale_image_and_label(
        processed_image, label, scale)

    # print(processed_image.shape)
    processed_image, label = random_crop_and_pad_image_and_labels(processed_image, label, (crop_height, crop_width))

    # print(processed_image.shape)

    cond_flip_lr = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)

    def orig(images, masks):
        return images, masks

    def flip(images, masks):
        return tf.map_fn(tf.image.flip_left_right, images), tf.map_fn(tf.image.flip_left_right, masks)

    processed_image, label = tf.cond(cond_flip_lr, lambda: flip(processed_image, label), lambda: orig(processed_image, label))

    rotate_rad = tf.random_uniform([], minval=-0.5, maxval=0.5)

    def rotate(processed_image, label, rad=0):
        return tf.contrib.image.rotate(processed_image, angles=rad), \
               tf.contrib.image.rotate(label, angles=rad)

    processed_image, label = rotate(processed_image, label, rotate_rad)
    processed_image.set_shape([image.shape[0], crop_height, crop_width, 3])
    label.set_shape([image.shape[0], crop_height, crop_width, 1])
    label = tf.cast(label, dtype=tf.int32)
    return processed_image, label


def create_inputs_fn(
        tfrecord_filenames: [str],
        num_epochs: int,
        batch_size: int,
        image_size: (int, int),
        scope: str,
        is_training: bool):
    """

    :param tfrecord_filenames:
    :param num_epochs:
    :param batch_size:
    :param image_size:
    :param scope:
    :param is_training:
    :return:
    """

    iter_init_hook = IteratorInitializerHook()

    def inputs_fn():
        with tf.variable_scope(scope):
            iterator = _create_initializable_iterator(tfrecord_filenames, num_epochs, batch_size, image_size)

            images, masks = iterator.get_next()

            if is_training:
                images, masks = _augment(
                    images,
                    masks,
                    crop_height=image_size[0],
                    crop_width=image_size[1],
                    min_resize_value=None,
                    max_resize_value=None,
                    resize_factor=None,
                    min_scale_factor=1.0,
                    max_scale_factor=1.3,
                    scale_factor_step_size=0.1)

            iter_init_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)

        return images, masks

    return inputs_fn, iter_init_hook
