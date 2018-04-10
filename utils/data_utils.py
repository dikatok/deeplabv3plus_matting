import tensorflow as tf

from utils.preprocessing_utils import random_crop, random_flip_left_right, random_rescale, random_rotate


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        if self.iterator_initializer_func is None:
            raise Exception("IteratorInitializerHook.iterator_initializer_func is not assigned")
        self.iterator_initializer_func(session)


def _create_parse_record_fn(image_size: (int, int)):
    """Create parse_record_fn which is used to parse single example from tf records

    Arguments:
        image_size: desired size of image and label
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


def _create_one_shot_iterator(tfrecord_filenames: [str],
                              num_epochs: int,
                              batch_size: int,
                              image_size: (int, int),
                              shuffle_buffer_size: int,
                              is_training: bool):
    """Function to create one shot iterator

    Arguments:
        tfrecord_filenames: list of tfrecord filenames
        num_epochs: number of epochs to repeat
        batch_size: number of samples per batch
        image_size: desired image and label size

    Returns:
        one_shot_iterator which yield batches of (images, labels)
    """

    dataset = tf.data.TFRecordDataset(tfrecord_filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.map(_create_parse_record_fn(image_size))

    dataset = dataset.prefetch(batch_size)

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.batch(batch_size)

    dataset = dataset.map(lambda images, labels: _augment(
        images,
        labels,
        crop_height=image_size[0],
        crop_width=image_size[1],
        min_scale=0.7,
        max_scale=1.3))

    return dataset.make_one_shot_iterator()


def _create_initializable_iterator(tfrecord_filenames: [str],
                                   num_epochs: int,
                                   batch_size: int,
                                   image_size: (int, int),
                                   shuffle_buffer_size: int,
                                   is_training: bool):
    """Function to create initializable iterator

    Arguments:
        tfrecord_filenames: list of tfrecord filenames
        num_epochs: number of epochs to repeat
        batch_size: number of samples per batch
        image_size: desired image and label size

    Returns:
        initializable_iterator which yield batches of (images, labels)
    """

    dataset = tf.data.TFRecordDataset(tfrecord_filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.map(_create_parse_record_fn(image_size))

    dataset = dataset.prefetch(batch_size)

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.batch(batch_size)

    dataset = dataset.map(lambda images, labels: _augment(
        images,
        labels,
        crop_height=image_size[0],
        crop_width=image_size[1],
        min_scale=0.7,
        max_scale=1.3))

    return dataset.make_initializable_iterator()


def _augment(images,
             labels,
             crop_height,
             crop_width,
             min_scale=1.,
             max_scale=1.):

    images_shape = images.shape.as_list()

    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.float32)

    images, labels = random_rescale(images, labels, min_scale, max_scale)

    images, labels = random_crop(images, labels, crop_height, crop_width)

    images, labels = random_flip_left_right(images, labels)

    images, labels = random_rotate(images, labels, -0.5, 0.5)

    labels = tf.cast(labels, dtype=tf.int32)

    images.set_shape([images_shape[0], crop_height, crop_width, 3])
    labels.set_shape([images_shape[0], crop_height, crop_width, 1])

    return images, labels


def create_inputs_fn(tfrecord_filenames: [str],
                     num_epochs: int,
                     batch_size: int,
                     image_size: (int, int),
                     shuffle_buffer_size: int,
                     is_training: bool,
                     scope: str):
    """Create inputs_fn

    Arguments:
        tfrecord_filenames
        num_epochs
        batch_size
        image_size
        shuffle_buffer_size
        is_training: whether to augment the data or not
        scope

    Returns:
        inputs_fn which returns batches of (images, labels)
        init_hook: instance of SessionRunHook to initialize iterator on session created
    """

    iter_init_hook = IteratorInitializerHook()

    def inputs_fn():
        with tf.variable_scope(scope), tf.device("/cpu:0"):
            iterator = _create_initializable_iterator(
                tfrecord_filenames=tfrecord_filenames,
                num_epochs=num_epochs,
                batch_size=batch_size,
                image_size=image_size,
                shuffle_buffer_size=shuffle_buffer_size,
                is_training=is_training)

            images, labels = iterator.get_next()

            iter_init_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)

        return images, labels

    return inputs_fn, iter_init_hook
