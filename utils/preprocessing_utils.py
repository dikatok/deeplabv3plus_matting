import tensorflow as tf


def random_rescale(images,
                   labels,
                   min_scale,
                   max_scale):
    """Randomly rescale images and labels in single batch

    Arguments:
        images
        labels
        min_scale
        max_scale

    Returns:
        augmented_images
        augmented_labels
    """

    height = tf.to_float(tf.shape(images)[1])
    width = tf.to_float(tf.shape(images)[2])

    assert max_scale >= min_scale > 0 and max_scale > 0

    scale = tf.random_uniform([], minval=min_scale, maxval=max_scale, dtype=tf.float32)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    images = tf.image.resize_images(
        images,
        [new_height, new_width],
        method=tf.image.ResizeMethod.BILINEAR)

    labels = tf.image.resize_images(
        labels,
        [new_height, new_width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return images, labels


def random_crop(images,
                labels,
                crop_height,
                crop_width):
    """Randomly crop images and labels in single batch with necessary padding

    Arguments:
        images
        labels
        crop_height
        crop_width

    Returns:
        augmented_images
        augmented_labels
    """

    height = tf.shape(images)[1]
    width = tf.shape(images)[2]

    concat = tf.concat([images, labels], axis=3)

    concat_padded = tf.image.pad_to_bounding_box(
        concat,
        offset_height=0,
        offset_width=0,
        target_height=tf.maximum(crop_height, height),
        target_width=tf.maximum(crop_width, width))

    concat_cropped = tf.map_fn(lambda x: tf.random_crop(x, [crop_height, crop_width, 4]), concat_padded)

    images = concat_cropped[:, :, :, :3]
    labels = concat_cropped[:, :, :, 3:]

    return images, labels


def random_flip_left_right(images, labels):
    """Random horizontal flip images and labels in single batch (either all are flipped or all are not)

    Arguments:
        images
        labels

    Returns:
        augmented_images
        augmented_labels
    """

    cond_flip_lr = tf.less(tf.random_uniform([], 0, 1.0), .5)

    images = tf.cond(cond_flip_lr, lambda: tf.reverse(images, [2]), lambda: images)
    labels = tf.cond(cond_flip_lr, lambda: tf.reverse(labels, [2]), lambda: labels)

    return images, labels


def random_rotate(images,
                  labels,
                  min_rotate,
                  max_rotate):
    """Randomly rotate images and labels in single batch

    Arguments:
        images
        labels
        min_rotate (in rad)
        max_rotate (in rad)

    Returns:
        augmented_images
        augmented_labels
    """

    rotate_rad = tf.random_uniform([], minval=min_rotate, maxval=max_rotate)

    images = tf.contrib.image.rotate(images, angles=rotate_rad)
    labels = tf.contrib.image.rotate(labels, angles=rotate_rad)

    return images, labels
