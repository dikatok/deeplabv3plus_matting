import tensorflow as tf


def define_train_args():
    """

    :return:
    """

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
        default=50,
        help='Elapsed steps interval to save summaries.')

    tf.app.flags.DEFINE_integer(
        name='checkpoints_steps',
        default=1000,
        help='Elapsed steps interval to save checkpoints.')

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


def define_convert_saved_model_args():
    """

    :return:
    """

    tf.app.flags.DEFINE_string(
        name='model_dir',
        default='./ckpts',
        help='Directory to save checkpoints.')

    tf.app.flags.DEFINE_string(
        name='saved_model_dir',
        default='./saved_model',
        help='Directory to save checkpoints.')

    tf.app.flags.DEFINE_integer(
        name='image_size',
        default=299,
        help='Input image size.')

    tf.app.flags.DEFINE_string(
        name='backbone',
        default='xception',
        help='Input image size.')
