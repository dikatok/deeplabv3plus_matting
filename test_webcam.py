import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS

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
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(15, 0.1)

    try:
        graph = _load_graph(FLAGS.frozen_model_path)

        with tf.Session(graph=graph) as sess:
            inputs = graph.get_tensor_by_name('graph/images:0')
            outputs = tf.image.resize_bilinear(
                tf.cast(graph.get_tensor_by_name('graph/predictions:0'), dtype=tf.float32),
                size=tf.shape(inputs)[1:3]) * inputs

            while cap.isOpened():
                ret, frame = cap.read()

                frame = cv2.flip(frame, 1)

                if len(frame.shape) < 3:
                    break

                result = sess.run(outputs, {inputs: [frame]})

                cv2.imshow('result', result[0] / 255.)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()


if __name__ == "__main__":
    tf.app.run(main=main)
