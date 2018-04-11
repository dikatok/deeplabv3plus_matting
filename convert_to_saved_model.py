import tensorflow as tf

from models.model import create_model_fn
from utils.train_utils import create_estimator_fn

FLAGS = tf.app.flags.FLAGS

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
    default=128,
    help='Input image size.')

tf.app.flags.DEFINE_string(
    name='backbone',
    default='xception',
    help='Input image size.')


def main(_):
    image_size = (FLAGS.image_size, FLAGS.image_size)

    model_fn = create_model_fn(
        backbone=FLAGS.backbone,
        output_stride=16,
        img_size=image_size)

    estimator_fn = create_estimator_fn(
        model_fn=model_fn,
        loss_fn=None,
        metrics_fn=None)

    estimator = tf.estimator.Estimator(
        model_fn=estimator_fn,
        model_dir=FLAGS.model_dir,
        params={})

    def serving_input_fn():
        images = tf.placeholder(
            shape=[None, None, None, 3],
            dtype=tf.float32,
            name="images")
        images = tf.image.resize_bilinear(images, size=image_size)
        return tf.estimator.export.TensorServingInputReceiver(images, images)

    estimator.export_savedmodel(FLAGS.saved_model_dir, serving_input_fn)


if __name__ == "__main__":
    tf.app.run(main=main)
