import tensorflow as tf


def chamfer_similarity(sim, max_axis=1, mean_axis=0):
    sim = tf.reduce_max(input_tensor=sim, axis=max_axis, keepdims=True)
    sim = tf.reduce_mean(input_tensor=sim, axis=mean_axis, keepdims=True)
    return tf.squeeze(sim, [max_axis, mean_axis])


def symmetric_chamfer_similarity(sim, axes=[0, 1]):
    return (chamfer_similarity(sim, axes[0], axes[1]) +
            chamfer_similarity(sim, axes[1], axes[0])) / 2


def triplet_loss(sim_pos, sim_neg, gamma=0.5):
    with tf.compat.v1.variable_scope('triplet_loss'):
        return tf.maximum(0., sim_neg - sim_pos + gamma)


def similarity_regularization_loss(sim, lower_limit=-1., upper_limit=1.):
    with tf.compat.v1.variable_scope('similarity_regularization_loss'):
        return tf.reduce_sum(input_tensor=tf.abs(tf.minimum(.0, sim - lower_limit))) + \
               tf.reduce_sum(input_tensor=tf.abs(tf.maximum(.0, sim - upper_limit)))
