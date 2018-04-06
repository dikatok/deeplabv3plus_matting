import tensorflow as tf
import numpy as np
from typing import Union


def create_train_op(
        loss: tf.Tensor,
        learning_rate: Union[tf.Tensor, float],
        momentum: Union[tf.Tensor, float]) -> tf.Operation:
    """Create train op

    Args:
        loss (tf.Tensor): Loss tensor
        learning_rate (float): Initial learning rate or decaying learning rate tensor
        momentum (float): Momentum value

    Returns:
        train op
    """

    global_step = tf.train.get_global_step()

    with tf.variable_scope("optimizer"):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum,
            use_nesterov=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)

    return train_op


def create_estimator_fn(
        model_fn,
        loss_fn,
        metrics_fn):

    def estimator_fn(features, labels, mode, params):

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        tf.keras.backend.set_learning_phase(is_training)

        images = tf.identity(features, name="images")
        labels = tf.identity(labels, name="labels")

        global_step = tf.train.get_or_create_global_step()

        logits = tf.identity(model_fn(images), name="logits")

        with tf.variable_scope("predictions"):
            preds = tf.identity(tf.expand_dims(tf.argmax(logits, axis=3), axis=3), name="predictions")

        loss = loss_fn(logits=logits, labels=labels, weight_decay=params.weight_decay)

        metrics_ops = metrics_fn(predictions=preds, labels=labels)

        if is_training:
            learning_rate = tf.train.polynomial_decay(
                learning_rate=params.learning_rate,
                global_step=global_step,
                decay_steps=params.learning_rate_decay_steps,
                end_learning_rate=params.end_learning_rate,
                power=params.learning_rate_decay)
            tf.summary.scalar("learning_rate", learning_rate)
            train_op = create_train_op(loss, learning_rate, params.momentum)
        else:
            train_op = None

        summary_prefix = "train"
        if not is_training:
            summary_prefix = "eval"

        tf.summary.image(f"{summary_prefix}_images", images)
        tf.summary.image(f"{summary_prefix}_labels", tf.cast(tf.scalar_mul(255, labels), dtype=tf.uint8))
        tf.summary.image(f"{summary_prefix}_predictions", tf.cast(preds, dtype=tf.float32) * images)

        tf.summary.scalar(f"{summary_prefix}_loss", loss)

        tf.summary.scalar(f"{summary_prefix}_accuracy", metrics_ops["accuracy"][1])
        tf.summary.scalar(f"{summary_prefix}_iou", metrics_ops["iou"][1])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=preds,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics_ops)

    return estimator_fn
