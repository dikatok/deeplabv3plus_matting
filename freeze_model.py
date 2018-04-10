import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str)
    parser.add_argument('--ckpt_iter', default=-1, type=int)
    parser.add_argument('--output_graph', default="model.pb", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()

    ckpt_path = tf.train.latest_checkpoint(args.ckpt_dir)
    if args.ckpt_iter >= 0:
        ckpt_path = os.path.join(args.ckpt_dir, "ckpt-{iter}".format(iter=args.ckpt_iter))

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["serve"], "./saved_model/1523316296")
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("images:0")
        model = graph.get_tensor_by_name("predictions:0")

        convert_variables_to_constants = tf.graph_util.convert_variables_to_constants
        output_graph_def = convert_variables_to_constants(sess,  tf.get_default_graph().as_graph_def(),
                                                          ["predictions"])

        input_names = ["images"]
        output_names = ["predictions"]
        transforms = ["strip_unused_nodes", "fold_batch_norms", "fold_constants", "quantize_weights"]
        # transforms =[]
        transformed_graph_def = TransformGraph(output_graph_def, input_names,
                                               output_names, transforms)

        with tf.gfile.GFile(args.output_graph, "wb") as f:
            f.write(transformed_graph_def.SerializeToString())
