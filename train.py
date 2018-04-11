import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.contrib import training as tf_training

import shutil

from models.model import create_model_fn
from models.losses import create_loss_fn
from models.metrics import create_metrics_fn

from utils.train_utils import create_estimator_fn
from utils.data_utils import create_inputs_fn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    name='model_dir',
    default='./ckpts',
    help='Directory to save checkpoints.')

tf.app.flags.DEFINE_bool(
    name='restart_training',
    default=False,
    help='Restart from step 1 and remove summaries and checkpoints.')

tf.app.flags.DEFINE_integer(
    name='train_epochs',
    default=50,
    help='Number of training epochs')

tf.app.flags.DEFINE_integer(
    name='batch_size',
    default=16,
    help='Number of examples per batch.')

tf.app.flags.DEFINE_integer(
    name='image_size',
    default=299,
    help='Input image size.')

tf.app.flags.DEFINE_string(
    name='backbone',
    default='xception',
    help='Encoder backbone architecture (xception or mobilenetv2).')

tf.app.flags.DEFINE_list(
    name='train_files',
    default=['./train-00001-of-00001'],
    help='List of training tfrecord filenames')

tf.app.flags.DEFINE_integer(
    name='summary_steps',
    default=10,
    help='Elapsed steps interval to save summaries.')

tf.app.flags.DEFINE_integer(
    name='val_epoch_interval',
    default=2,
    help='The number of training epochs to run between evaluations.')

tf.app.flags.DEFINE_list(
    name='val_files',
    default=['./val-00001-of-00001'],
    help='List of validation tfrecord filenames')

tf.app.flags.DEFINE_float(
    name='learning_rate',
    default=0.05,
    help='Initial learning rate.')

tf.app.flags.DEFINE_float(
    name='end_learning_rate',
    default=1e-6,
    help='Final learning rate.')

tf.app.flags.DEFINE_float(
    name='learning_rate_decay',
    default=0.94,
    help='Learning rate decay rate.')

tf.app.flags.DEFINE_float(
    name='momentum',
    default=0.9,
    help='SGD momentum value.')

tf.app.flags.DEFINE_float(
    name='weight_decay',
    default=4e-5,
    help='L2 regularization weight decay.')


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
        img_size=image_size,
        output_stride=16)

    loss_fn = create_loss_fn()

    metrics_fn = create_metrics_fn()

    estimator_fn = create_estimator_fn(
        model_fn=model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn)

    config = tf.estimator.RunConfig(
        tf_random_seed=42,
        save_summary_steps=FLAGS.summary_steps,
        save_checkpoints_steps=num_train_samples // FLAGS.batch_size,
        log_step_count_steps=FLAGS.summary_steps,
        model_dir=FLAGS.model_dir)

    params = tf_training.HParams(
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay=FLAGS.learning_rate_decay,
        end_learning_rate=FLAGS.end_learning_rate,
        # learning_rate_decay_steps=2000,
        learning_rate_decay_steps=2 * FLAGS.train_epochs * num_train_samples // FLAGS.batch_size,
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
        shuffle_buffer_size=num_train_samples,
        scope="train_inputs",
        is_training=True)

    val_inputs_fn, val_init_hook = create_inputs_fn(
        tfrecord_filenames=FLAGS.val_files,
        num_epochs=1,
        batch_size=FLAGS.batch_size,
        image_size=image_size,
        shuffle_buffer_size=num_val_samples,
        scope="val_inputs",
        is_training=False)

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
        start_delay_secs=120,
        throttle_secs=1200
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.app.run(main=main)
