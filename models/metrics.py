import tensorflow as tf


def iou_metric_fn(labels,
                  predictions):
    """Custom iou metrics

    The diff from tf.metrics.mean_iou is this fn only calculates iou for foreground (predictions == labels == 1)

    Arguments:
        labels
        predictions (argmax output)

    Returns:
        iou
        update_op
    """

    preds_ones = tf.equal(predictions, 1)
    labels_ones = tf.equal(labels, 1)
    i = tf.cast(tf.logical_and(preds_ones, labels_ones), dtype=tf.float32)
    u = tf.cast(tf.logical_or(preds_ones, labels_ones), dtype=tf.float32)
    iou = tf.reduce_sum(i, axis=[1, 2, 3]) / (tf.reduce_sum(u, axis=[1, 2, 3]) + 1e-6)

    iou, update_iou_op = tf.metrics.mean(iou)

    return iou, update_iou_op


def create_metrics_fn():
    """First order function which returns metrics_fn

    Returns:
        metrics_fn with signature (predictions, labels) and returns dict of iou and acc
    """

    def metrics_fn(predictions, labels):
        with tf.variable_scope("iou"):
            iou = iou_metric_fn(predictions=predictions, labels=labels)

        with tf.variable_scope("accuracy"):
            accuracy = tf.metrics.accuracy(predictions=predictions, labels=labels)

        return {
            "iou": iou,
            "accuracy": accuracy
        }

    return metrics_fn
