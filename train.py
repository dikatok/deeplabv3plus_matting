import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.contrib import training as tf_training
import shutil
from models.model import create_model_fn
from models.losses import create_loss_fn
from models.metrics import create_metrics_fn
from args import define_train_args
from utils.train_utils import create_estimator_fn
from utils.data_utils import create_inputs_fn

FLAGS = tf.app.flags.FLAGS

define_train_args()


def main(_):
    if FLAGS.restart_training:
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

    image_size = (FLAGS.image_size, FLAGS.image_size)

    num_train_samples = sum(1 for f in file_io.get_matching_files(FLAGS.train_files)
                            for _ in tf.python_io.tf_record_iterator(f))

    num_val_samples = sum(1 for f in file_io.get_matching_files(FLAGS.val_files)
                          for _ in tf.python_io.tf_record_iterator(f))

    print(f"Number of training samples: {num_train_samples}")
    print(f"Number of validation samples: {num_val_samples}")

    model_fn = create_model_fn(
        backbone=FLAGS.backbone,
        img_size=image_size)

    loss_fn = create_loss_fn()

    metrics_fn = create_metrics_fn()

    estimator_fn = create_estimator_fn(
        model_fn=model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn)

    config = tf.estimator.RunConfig(
        tf_random_seed=42,
        save_summary_steps=FLAGS.summary_steps,
        # save_checkpoints_steps=FLAGS.checkpoints_steps,
        save_checkpoints_steps=num_train_samples // FLAGS.batch_size,
        log_step_count_steps=FLAGS.summary_steps,
        model_dir=FLAGS.model_dir)

    params = tf_training.HParams(
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay=FLAGS.learning_rate_decay,
        end_learning_rate=FLAGS.end_learning_rate,
        learning_rate_decay_steps=FLAGS.train_epochs * num_train_samples // FLAGS.batch_size,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay)

    estimator = tf.estimator.Estimator(
        model_fn=estimator_fn,
        params=params,
        config=config)

    train_inputs_fn, train_init_hook = create_inputs_fn(
        tfrecord_filenames=FLAGS.train_files,
        num_epochs=FLAGS.val_epoch_interval,
        batch_size=FLAGS.batch_size,
        image_size=image_size,
        scope="train_inputs",
        is_training=True)

    val_inputs_fn, val_init_hook = create_inputs_fn(
        tfrecord_filenames=FLAGS.val_files,
        num_epochs=1,
        batch_size=FLAGS.batch_size,
        image_size=image_size,
        scope="val_inputs",
        is_training=False)

    def serving_input_fn():
        images = tf.placeholder(shape=[None, image_size[0], image_size[1], 3], dtype=tf.float32)
        return tf.estimator.export.TensorServingInputReceiver(images, images)

    exporter = tf.estimator.LatestExporter(
        name='Servo',
        serving_input_receiver_fn=serving_input_fn,
        assets_extra=None,
        as_text=False,
        exports_to_keep=5)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_inputs_fn,
        max_steps=FLAGS.train_epochs * num_train_samples // FLAGS.batch_size,
        hooks=[train_init_hook]
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=val_inputs_fn,
        steps=None,
        name=None,
        hooks=[val_init_hook],
        exporters=None,  # Iterable of Exporters, or single one or None.
        start_delay_secs=120,
        throttle_secs=600
    )
    # estimator.evaluate(val_inputs_fn, steps=None, hooks=[val_init_hook])
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    # tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run(main=main)


