import tensorflow as tf
import cv2
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    name='input_path',
    default=None,
    help='Path to input video.')

tf.app.flags.DEFINE_string(
    name='output_path',
    default=None,
    help='Path to output video.')

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
    assert FLAGS.input_path is not None
    assert FLAGS.output_path is not None

    cap = cv2.VideoCapture(FLAGS.input_path)

    if cap.isOpened():
        width = cap.get(3)
        height = cap.get(4)

    save = cv2.VideoWriter(
        FLAGS.output_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        30.0,
        (int(width), int(height)),
        isColor=True)

    try:
        graph = _load_graph(FLAGS.frozen_model_path)

        with tf.Session(graph=graph) as sess:
            inputs = graph.get_tensor_by_name('graph/images:0')
            outputs = tf.image.resize_bilinear(
                tf.cast(graph.get_tensor_by_name('graph/predictions:0'), dtype=tf.float32),
                size=tf.shape(inputs)[1:3]) * inputs

            while cap.isOpened():
                ret, frame = cap.read()

                if frame is None:
                    break

                result = sess.run(outputs, {inputs: [frame]})

                save.write(np.uint8(result[0]))
    finally:
        cap.release()
        save.release()


if __name__ == "__main__":
    tf.app.run(main=main)
