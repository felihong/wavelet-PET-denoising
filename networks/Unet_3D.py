import numpy as np
from keras import backend as K
K.set_image_data_format("channels_first")
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, Deconvolution3D
from keras.layers.merge import concatenate


def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation=None):
    """
    Builds the 3D UNet Keras model using stride.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param batch_normalization: Boolean value indicating whether to include BN, by default set to False.
    :param activation: Activation function, use ReLu if not set.
    :return: Untrained 3D UNet model.
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    # Add levels with max pooling
    for layer_depth in range(depth): 
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                        strides=(1, 1, 1), batch_normalization=batch_normalization, activation=activation)
        if layer_depth < depth - 1: 
            layer2 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth)*2,
                                        strides=(2, 2, 2), batch_normalization=batch_normalization, activation=activation)
        else:
            layer2 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth)*2,
                                        strides=(1, 1, 1), batch_normalization=batch_normalization, activation=activation)
        current_layer = layer2
        levels.append([layer1, layer2])
    # Add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][0]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][0]._keras_shape[1],
                                                 input_layer=concat, 
                                                 batch_normalization=batch_normalization, activation=activation)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][0]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization, activation=activation)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    model = Model(inputs=inputs, outputs=final_convolution)
    return model


def unet_model_3d_maxPooling(input_shape, pool_size=(2, 2, 2), n_labels=1, deconvolution=False,
                            depth=4, n_base_filters=32, batch_normalization=False, activation=None):
    """
    Builds the 3D UNet Keras model using maxPooling.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param batch_normalization: Boolean value indicating whether to include BN, by default set to False.
    :param activation: Keras activation layer.
    :return: Untrained 3D UNet model.
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    # Add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, activation=activation)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, activation=activation)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
    # Add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, 
                                                 batch_normalization=batch_normalization, activation=activation)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization, activation=activation)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    model = Model(inputs=inputs, outputs=final_convolution)
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization, activation, 
                            kernel=(3, 3, 3), padding='same', strides=(1, 1, 1), instance_normalization=False):
    """Generate convolution blocks."""
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param image_shape: shape of the 3d image.
    :return: 5D vector of the shape of the output node.
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)