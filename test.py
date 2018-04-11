import tensorflow as tf

from glob import glob
import os


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    name='input_dir',
    default=None,
    help='Directory to be processed.')

tf.app.flags.DEFINE_string(
    name='output_dir',
    default=None,
    help='Directory of output.')

tf.app.flags.DEFINE_string(
    name='frozen_model_path',
    default='./model.pb',
    help='Path to save frozen model.')


def _load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="graph")

    return graph


def main(_):
    assert os.path.exists(FLAGS.input_dir)
    assert os.path.exists(FLAGS.frozen_model_path)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    graph = _load_graph(FLAGS.frozen_model_path)

    with tf.Session(graph=graph) as sess:

        inputs = graph.get_tensor_by_name('graph/images:0')
        outputs = tf.image.resize_bilinear(
            tf.cast(graph.get_tensor_by_name('graph/predictions:0'), dtype=tf.float32),
            size=tf.shape(inputs)[1:3]) * inputs

        for f in glob(os.path.join(FLAGS.input_dir, "*")):
            img = tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(f))

            result = sess.run(outputs, {inputs: [img]})

            tf.keras.preprocessing.image.array_to_img(result[0])\
                .save(os.path.join(FLAGS.output_dir, os.path.split(f)[-1]))


if __name__ == "__main__":
    tf.app.run(main=main)