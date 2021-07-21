"""
Mathias Perslev
MSc Bioinformatics

University of Copenhagen
November 2017
"""

from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.utils.conv_arithmetics import compute_receptive_fields

from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D,SpatialDropout2D,Add,Activation,Multiply
import numpy as np


class UNet(Model):
    """
    2D UNet implementation with batch normalization and complexity factor adj.

    See original paper at http://arxiv.org/abs/1505.04597
    """
    def __init__(self, n_classes, img_rows=None, img_cols=None, dim=None,
                 n_channels=1, depth=5, out_activation="softmax",
                 activation="relu", kernel_size=3, padding="same",
                 complexity_factor=1, l2_reg=None, logger=None, **kwargs):
        """
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        img_rows, img_cols (int, int):
            Image dimensions. Note that depending on image dims cropping may
            be necessary. To avoid this, use image dimensions DxD for which
            D * (1/2)^n is an integer, where n is the number of (2x2)
            max-pooling layers; in this implementation 4.
            For n=4, D \in {..., 192, 208, 224, 240, 256, ...} etc.
        dim (int):
            img_rows and img_cols will both be set to 'dim'
        n_channels (int):
            Number of channels in the input image.
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        out_activation (string):
            Activation function of output 1x1 conv layer. Usually one of
            'softmax', 'sigmoid' or 'linear'.
        activation (string):
            Activation function for convolution layers
        kernel_size (int):
            Kernel size for convolution layers
        padding (string):
            Padding type ('same' or 'valid')
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            2D convolution layer instead of default N.
        l2_reg (float in [0, 1])
            L2 regularization on Conv2D weights
        logger (MultiPlanarUNet.logging.Logger | ScreenLogger):
            MutliViewUNet.Logger object, logging to files or screen.
        """
        super(UNet, self).__init__()
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_col or dim")
        if dim:
            img_rows, img_cols = dim, dim

        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()

        # Set various attributes
        self.img_shape = (img_rows, img_cols, n_channels)
        self.n_classes = n_classes
        self.cf = np.sqrt(complexity_factor)
        self.kernel_size = kernel_size
        self.activation = activation
        self.out_activation = out_activation
        self.l2_reg = l2_reg
        self.padding = padding
        self.depth = depth
        self.n_base_filters=16
        self.n_segmentation_levels=4
      

        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field
        names = [x.__class__.__name__ for x in self.layers]
        index = names.index("UpSampling2D")
        self.receptive_field = compute_receptive_fields(self.layers[:index])[-1][-1]

        # Log the model definition
        self.log()

    def _create_encoder(self, in_, init_filters, kernel_reg=None,
                        name="encoder"):
        filters = init_filters
        residual_connections = []
        for i in range(self.depth):
            l_name = name + "_L%i" % i
            conv = Conv2D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv1")(in_)
            conv = Conv2D(int(filters * self.cf)*2, self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv2")(conv)
            bn = BatchNormalization(name=l_name + "_BN")(conv)
            in_ = MaxPooling2D(pool_size=(2, 2), name=l_name + "_pool")(bn)

            # Update filter count and add bn layer to list for residual conn.
            filters *= 2
            residual_connections.append(bn)
        return in_, residual_connections, filters

    def _create_bottom(self, in_, filters, kernel_reg=None, name="bottom"):
        conv = Conv2D(int(filters * self.cf), self.kernel_size,
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kernel_reg,
                      name=name + "_conv1")(in_)
        conv = Conv2D(int(filters * self.cf), self.kernel_size,
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kernel_reg,
                      name=name + "_conv2")(conv)
        bn = BatchNormalization(name=name + "_BN")(conv)
        return bn

    def _create_upsample(self, in_, res_conns, filters, kernel_reg=None,
                         name="upsample"):
        residual_connections = res_conns[::-1]
        for i in range(self.depth):
            l_name = name + "_L%i" % i
            # Reduce filter count
            filters /= 2

            # Up-sampling block
            # Note: 2x2 filters used for backward comp, but you probably
            # want to use 3x3 here instead.
            up = UpSampling2D(size=(2, 2), name=l_name + "_up")(in_)
            conv = Conv2D(int(filters * self.cf), 2,
                          activation=self.activation,
                          padding=self.padding, kernel_regularizer=kernel_reg,
                          name=l_name + "_conv1")(up)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)

            # Crop and concatenate
            cropped_res = self.crop_nodes_to_match(residual_connections[i], bn)
            merge = Concatenate(axis=-1,
                                name=l_name + "_concat")([cropped_res, bn])

            conv = Conv2D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv2")(merge)
            conv = Conv2D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv3")(conv)
            in_ = BatchNormalization(name=l_name + "_BN2")(conv)
        return in_

    def init_model(self):
        """
        Build the UNet model with the specified input image shape.
        """
        #inputs = Input(shape=self.img_shape)

        # Apply regularization if not None or 0
        #kr = regularizers.l2(self.l2_reg) if self.l2_reg else None

        """
        Encoding path
        """
        #in_, residual_cons, filters = self._create_encoder(in_=inputs,
                                                           #init_filters=16,
                                                           #kernel_reg=kr)

        """
        Bottom (no max-pool)
        """
        #bn = self._create_bottom(in_, filters, kr)

        """
        Up-sampling
        """
        #bn = self._create_upsample(bn, residual_cons, filters, kr)

        """
        Output modeling layer
        """
        #out = Conv2D(self.n_classes, 1, activation=self.out_activation)(bn)
        inputs = Input(self.img_shape)

        current_layer = inputs
        level_output_layers = list()
        level_filters = list()
        for level_number in range(self.depth):
            n_level_filters = (2**level_number) * self.n_base_filters
            level_filters.append(n_level_filters)

            if current_layer is inputs:
                in_conv = create_convolution_block(current_layer, n_level_filters)
            else:
                in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2))

            context_output_layer = create_context_module(in_conv, n_level_filters)

            summation_layer = Add()([in_conv, context_output_layer])
            level_output_layers.append(summation_layer)
            current_layer = summation_layer

        segmentation_layers = list()
        for level_number in range(self.depth - 2, -1, -1):
            up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
            attn= AttnGatingBlock(level_output_layers[level_number], up_sampling , level_filters[level_number]//2)
            concatenation_layer = Concatenate(axis=-1)([attn, up_sampling])
            localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
            current_layer = localization_output
            if level_number < self.n_segmentation_levels:
                segmentation_layers.insert(0, Conv2D(self.n_classes, (1, 1))(current_layer))

        output_layer = None
        for level_number in reversed(range(self.n_segmentation_levels)):
            segmentation_layer = segmentation_layers[level_number]
            if output_layer is None:
                output_layer = segmentation_layer
            else:
                output_layer = Add()([output_layer, segmentation_layer])

            if level_number > 0:
                output_layer = UpSampling2D(size=(2, 2))(output_layer)

        activation_block = Activation(self.out_activation)(output_layer)


        return [inputs], [activation_block)]
    def AttnGatingBlock(self,x, g, inter_shape):
        theta_x = Conv2D(inter_shape, (1, 1), strides=(1, 1), padding='same')(x)
        theta_x = BatchNormalization(axis=-1)(theta_x)
        phi_g = Conv2D(inter_shape, (1, 1),strides=(1, 1), padding='same')(g)
        phi_g = BatchNormalization(axis=-1)(phi_g)
        concat_xg = add([phi_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = Conv2D(1, (1, 1), padding='same')(act_xg)
        psi = BatchNormalization(axis=-1)(psi)
        psi = Activation('sigmoid')(psi)
        y = Multiply([psi, x])
    return y
    def create_convolution_block(self,input_layer, n_filters, batch_normalization=True, kernel=(3, 3), activation=None,
                             padding='same', strides=(1, 1), instance_normalization=False):
        """

        :param strides:
        :param input_layer:
        :param n_filters:
        :param batch_normalization:
        :param kernel:
        :param activation: Keras activation layer to use. (default is 'relu')
        :param padding:
        :return:
        """
        layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
        if batch_normalization:
            layer = BatchNormalization(axis=-1)(layer)
        elif instance_normalization:
            try:
                from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
            except ImportError:
                raise ImportError("Install keras_contrib in order to use instance normalization."
                                  "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
            layer = InstanceNormalization(axis=-1)(layer)
        if activation is None:
            return Activation('relu')(layer)
        else:
            return activation()(layer)

    def create_localization_module(self,input_layer, n_filters):
        convolution1 = create_convolution_block(input_layer, n_filters)
        convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1))
        return convolution2


    def create_up_sampling_module(self,input_layer, n_filters, size=(2, 2)):
        up_sample = UpSampling2D(size=size)(input_layer)
        convolution = create_convolution_block(up_sample, n_filters)
        return convolution


    def create_context_module(self,input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last"):
        convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
        dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(convolution1)
        convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
        return convolution2

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-1]
        s2 = np.array(node2.get_shape().as_list())[1:-1]

        if np.any(s1 != s2):
            c = (s1 - s2).astype(np.int)
            cr = np.array([c//2, c//2]).T
            cr[:, 1] += c % 2
            cropped_node1 = Cropping2D(cr)(node1)
            self.label_crop += cr
        else:
            cropped_node1 = node1

        return cropped_node1

    def log(self):
        self.logger("UNet Model Summary\n------------------")
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])
        self.logger("N classes:         %i" % self.n_classes)
        self.logger("CF factor:         %.3f" % self.cf**2)
        self.logger("Depth:             %i" % self.depth)
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Out activation:    %s" % self.out_activation)
        self.logger("Receptive field:   %s" % self.receptive_field)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)
        self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))
