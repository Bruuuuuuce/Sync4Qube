import tensorflow as tf

#===================================================================================================
#===================================================================================================
'''Utils'''


def _adjust_labels(labels, predictions):
    '''Adjust the 'labels' tensor by squeezing it if needed.'''
    labels = tf.cast(labels, tf.int32)
    if len(predictions.shape) == len(labels.shape):
        labels = tf.squeeze(labels, [-1])
    return labels, predictions


def _validate_rank(labels, predictions, weights):
    if weights is not None and len(weights.shape) != len(labels.shape):
        raise RuntimeError((
            "Weight and label tensors were not of the same rank. weights.shape "
            "was %s, and labels.shape was %s.") %
                           (predictions.shape, labels.shape))
    if (len(predictions.shape) - 1) != len(labels.shape):
        raise RuntimeError((
            "Expects `labels` to have a "
            "rank of one less than `predictions`. labels.shape was %s, and "
            "predictions.shape was %s.") % (labels.shape, predictions.shape))


#===================================================================================================
#===================================================================================================
'''Regression losses'''


def y_weighted_mse(y_true, y_pred):
    y_true, y_pred = _adjust_labels(y_true, y_pred)
    err_weighted = (y_true - y_pred)**2 * (tf.maximum(y_true**2, 1e-7))
    mean_err_weighted = tf.reduce_mean(err_weighted)
    return mean_err_weighted


def MSPE(y_true, y_pred):
    y_true, y_pred = _adjust_labels(y_true, y_pred)
    err_percent = (y_true - y_pred)**2 / (tf.maximum(y_true**2, 1e-7))
    mean_err_percent = tf.reduce_mean(err_percent)
    return mean_err_percent

def rank_penalty_mse(y_true, y_pred, alpha=0.05, beta=1.0):
    mse = tf.math.square(y_pred - y_true)
    penalty = tf.math.maximum(0., -4 * tf.math.square(y_pred - 0.5) + beta)
    loss = mse + alpha * penalty
    return loss

def fix_weighted_mse(y_true, y_pred, low=0.5, high=1.0, alpha=0.5):
    weights = tf.where(y_true > low, 1.0, alpha)
    loss = weights * tf.square(y_true - y_pred)

    return loss

def fix_weighted_mae(y_true, y_pred, low=0.5, high=1.0, alpha=0.5):
    weights = tf.where(y_true > low, 1.0, alpha)
    loss = weights * tf.abs(y_true - y_pred)

    return loss


#===================================================================================================
#===================================================================================================
'''Weighted sparse categorical cross-entropy loss'''


def weighted_sparse_categorical_crossentropy_loss(labels,
                                                  predictions,
                                                  weights=None,
                                                  from_logits=False):
    # When using these functions with the Keras core API, we will need to squeeze
    # the labels tensor - Keras adds a spurious inner dimension.
    labels, predictions = _adjust_labels(labels, predictions)
    _validate_rank(labels, predictions, weights)

    example_losses = tf.keras.losses.sparse_categorical_crossentropy(
        labels, predictions, from_logits=from_logits)

    if weights is None:
        return tf.reduce_mean(example_losses)
    weights = tf.cast(weights, predictions.dtype)
    return tf.math.divide_no_nan(tf.reduce_sum(example_losses * weights),
                                 tf.reduce_sum(weights))
