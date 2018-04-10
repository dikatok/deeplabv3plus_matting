import tensorflow as tf

from args import define_convert_saved_model_args

from models.model import create_model_fn
from utils.train_utils import create_estimator_fn

FLAGS = tf.app.flags.FLAGS

define_convert_saved_model_args()


def main(_):
    image_size = (FLAGS.image_size, FLAGS.image_size)

    model_fn = create_model_fn(
        backbone=FLAGS.backbone,
        output_stride=8,
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
            shape=[None, image_size[0], image_size[1], 3],
            dtype=tf.float32,
            name="images")
        return tf.estimator.export.TensorServingInputReceiver(images, images)

    estimator.export_savedmodel(FLAGS.saved_model_dir, serving_input_fn)


if __name__ == "__main__":
    tf.app.run(main=main)
