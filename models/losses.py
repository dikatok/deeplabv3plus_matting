import tensorflow as tf


def create_loss_fn():
    def loss_fn(logits, labels, weight_decay):
        with tf.variable_scope("total_loss"):
            with tf.variable_scope("cross_entropy"):
                cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

            with tf.variable_scope("l2_reg"):
                l2_reg = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

            total_loss = cross_entropy_loss + l2_reg

        return total_loss

    return loss_fn
