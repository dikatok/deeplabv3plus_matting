import tensorflow as tf

from typing import Union


def create_train_op(
        loss: tf.Tensor,
        learning_rate: Union[tf.Tensor, float],
        momentum: Union[tf.Tensor, float]):
    """Create training op

    Arguments:
        loss: loss tensor
        learning_rate: either tensor or float
        momentum
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
    """First order function which returns estimator_fn

    Arguments:
        model_fn: function to create computation graph and get model output
        loss_fn: function to get loss tensor
        metrics_fn: function to get metrics dict

    Returns:
        estimator_fn with signature (features, labels, mode, params) and returns EstimatorSpec
    """

    def estimator_fn(features, labels, mode, params):

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        tf.keras.backend.set_learning_phase(is_training)

        images = tf.identity(features, name="images")

        logits = tf.identity(model_fn(images, is_training=is_training), name="logits")

        preds = tf.identity(tf.expand_dims(tf.argmax(logits, axis=3), axis=3), name="predictions")

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=preds,
                export_outputs={
                    "predictions": tf.estimator.export.PredictOutput(preds)
                })

        labels = tf.identity(labels, name="labels")

        loss = loss_fn(logits=logits, labels=labels, weight_decay=params.weight_decay)

        metrics_ops = metrics_fn(predictions=preds, labels=labels)

        global_step = tf.train.get_or_create_global_step()

        if is_training:
            learning_rate = tf.train.exponential_decay(
                learning_rate=params.learning_rate,
                global_step=global_step,
                decay_steps=200,
                decay_rate=params.learning_rate_decay,
                staircase=True
            )
            tf.summary.scalar("learning_rate", learning_rate)
            train_op = create_train_op(loss, params.learning_rate, params.momentum)
        else:
            train_op = None

        tf.summary.image("images", tf.concat([
            images,
            tf.tile(
                tf.cast(tf.scalar_mul(255, labels), dtype=tf.float32),
                multiples=tf.stack([1, 1, 1, 3])),
            tf.tile(
                tf.cast(tf.scalar_mul(255, preds), dtype=tf.float32),
                multiples=tf.stack([1, 1, 1, 3]))],
            axis=2))

        tf.summary.scalar("loss", loss)

        tf.summary.scalar("accuracy", metrics_ops["accuracy"][1])
        tf.summary.scalar("iou", metrics_ops["iou"][1])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=preds,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics_ops)

    return estimator_fn
