"""
 " License:
 " -----------------------------------------------------------------------------
 " Copyright (c) 2018, Ratnajit Mukherjee.
 " All rights reserved.
 "
 " Redistribution and use in source and binary forms, with or without
 " modification, are permitted provided that the following conditions are met:
 "
 " 1. Redistributions of source code must retain the above copyright notice,
 "    this list of conditions and the following disclaimer.
 "
 " 2. Redistributions in binary form must reproduce the above copyright notice,
 "    this list of conditions and the following disclaimer in the documentation
 "    and/or other materials provided with the distribution.
 "
 " 3. Neither the name of the copyright holder nor the names of its contributors
 "    may be used to endorse or promote products derived from this software
 "    without specific prior written permission.
 "
 " THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 " ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 " LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 " CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 " SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 " INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 " CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 " ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 " POSSIBILITY OF SUCH DAMAGE.
 " -----------------------------------------------------------------------------
 " Description: Residual Network customized for TinyImageNet project (pre-activation version)
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: October 2018
"""
# various imports
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers import concatenate, AveragePooling2D, add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import VarianceScaling
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K


class ResNet:
    def __init__(self):
        print('Loading Residual Network')

    def bn_conv2d(self, x, filter_size, kernel_size, padding_type, activation_type, strides=(1, 1)):
        """
        Batch Normalization and pre-activation followed by a convolution layer
        """
        weight = 1e-4
        # Batch norm
        x = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9)(x)
        # Pre-activation
        if activation_type == 'LeakyRelu':
            x = LeakyReLU(alpha=0.3)(x)
        else:
            x = Activation(activation_type)(x)

        # Convolution 2D
        x = Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, kernel_regularizer=l2(weight),
                   kernel_initializer="he_normal", padding=padding_type, activation='linear', use_bias=False)(x)
        return x

    def residual_module(self, x, filter_size, stride_size, reduce=False):
        shortcut = x
        # First bn_conv2d block
        conv1 = self.bn_conv2d(x, filter_size=int(filter_size * 0.25), kernel_size=1, padding_type='same', activation_type='linear')
        # Second bn_conv2d block (this block also reduces spatial resolution)
        conv2 = self.bn_conv2d(conv1, filter_size=int(filter_size * 0.25), kernel_size=3, padding_type='same', activation_type='linear', strides=stride_size)
        # Third
        conv3 = self.bn_conv2d(conv2, filter_size=filter_size, kernel_size=1, padding_type='same', activation_type='linear')

        if reduce is True:
            shortcut = self.bn_conv2d(x, filter_size, kernel_size=1, padding_type='same', activation_type='linear', strides=stride_size)

        x = add([conv3, shortcut])

        return x

    def resnet_build(self, input_shape, num_classes, stage_list, filter_list):
        input_layer = Input(input_shape)
        # First convolution block to capture larger maps
        x = Conv2D(filters=filter_list[0], kernel_size=(5, 5), kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4), padding='same', activation='linear')(input_layer)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        # loop over the number of stages and the number of filter to stack the residual modules

        for i in range(0, len(stage_list)):
            if i == 0:
                stride_size = (1, 1)
            else:
                stride_size = (2, 2)

            x = self.residual_module(x, filter_list[i + 1], stride_size=stride_size, reduce=True)

            for j in range(0, stage_list[i] - 1):
                x = self.residual_module(x, filter_list[i + 1], stride_size=(1, 1))

        # head of the network
        x = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = AveragePooling2D(pool_size=(8, 8))(x)
        x = Flatten()(x)
        x = Dense(units=num_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=x)
        return model


if __name__ == '__main__':
    input_shape = (64, 64, 3)
    output_classes = 200
    stage_list = (3, 4, 6)
    filter_list = (64, 128, 256, 512)
    resnet = ResNet()
    model = resnet.resnet_build(input_shape, output_classes, stage_list=stage_list, filter_list=filter_list)
    model.summary()
