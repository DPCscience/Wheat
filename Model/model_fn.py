"""Define the model."""

import tensorflow as tf

from Model.utils import Params, fixed_kernel_initializer, fixed_bias_initializer

def create_block(
    block_name, inputs, num_filters, params,
    kernel_size=3, padding='same', is_training=True,
    kernel_initializer=None, trainable=True, pool=True,
    bias_initializer=tf.zeros_initializer()):
    """
    Creates a single convolution block.
    """
    with tf.variable_scope(block_name):
        out = tf.layers.conv2d(
            inputs, num_filters, kernel_size,
            padding=padding, kernel_initializer=kernel_initializer,
            trainable=trainable, bias_initializer=bias_initializer
        )
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(
                out, momentum=params.bn_momentum,
                training=is_training
            )
        out = tf.nn.relu(out)
        if pool:
            out = tf.layers.max_pooling2d(out, 2, 2)
    return out


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)
    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters
    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

    num_filters = params.num_channels # Number of filters in the first conv block
    out = create_block(
        'conv_block_1', images, num_filters, params,
        trainable=False, kernel_initializer=fixed_kernel_initializer(0),
        bias_initializer=fixed_bias_initializer(0)
    )
    out = create_block(
        'conv_block_2', out, num_filters*2, params,
    )
    out = create_block(
        'conv_block_3', out, num_filters*4, params
    )
    # out = create_block(
    #     'conv_block_4', out, num_filters*2, params,
    # )
    assert out.get_shape().as_list() == [None, 32, 32, num_filters * 4]
    out = tf.reshape(out, [-1, 32 * 32 * num_filters * 4])

    with tf.variable_scope('fc1'):
        out = tf.layers.dense(out, num_filters*4)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(
                out, momentum=params.bn_momentum, training=is_training
            )
        out = tf.nn.relu(out)

    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(out, params.num_labels)
    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits
    )
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(labels, predictions), tf.float32)
    )

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec