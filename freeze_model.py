import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

import os
from glob import glob

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    name='saved_model_dir',
    default='./saved_model',
    help='Directory of saved model.')

tf.app.flags.DEFINE_string(
    name='output_path',
    default='./model.pb',
    help='Output path to save frozen model.')


def main(_):
    assert os.path.exists(FLAGS.saved_model_dir)

    saved_models = glob(os.path.join(FLAGS.saved_model_dir, '*'))

    assert len(saved_models) > 0

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_models[-1])

        graph = tf.get_default_graph()

        convert_variables_to_constants = tf.graph_util.convert_variables_to_constants

        output_graph_def = convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            ["predictions"])

        input_names = ["images"]
        output_names = ["predictions"]

        transforms = ["strip_unused_nodes", "fold_batch_norms", "fold_constants", "quantize_weights"]

        transformed_graph_def = TransformGraph(
            output_graph_def,
            input_names,
            output_names,
            transforms)

        with tf.gfile.GFile(FLAGS.output_path, "wb") as f:
            f.write(transformed_graph_def.SerializeToString())


if __name__ == "__main__":
    tf.app.run(main=main)
