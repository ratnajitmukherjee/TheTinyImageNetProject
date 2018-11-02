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
 " Description: Inception V4 Network customized for TinyImageNet project
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: October 2018
"""

from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers import concatenate, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import VarianceScaling
from keras.regularizers import l2
from keras.models import Model


class InceptionV4:
    def __init__(self):
        print('Loading Inception Network...')
    
# Define the convolution layer
    def conv2d_bn(self, x, filter_size, kernel_size, padding_type, activation_type, strides=(1, 1)):
        weight = 5e-4
        x = Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, kernel_regularizer=l2(weight),
                   kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                   padding=padding_type, activation='linear')(x)

        if activation_type == 'LeakyRelu':
            x = LeakyReLU(alpha=0.3)(x)
        else:
            x = Activation(activation_type)(x)
        x = BatchNormalization(axis=-1)(x)
        return x

    # Define the Maxpool Layers
    def maxpool_2d(self, x, pool_size, stride_size, padding_type):
        if stride_size is None:
            stride_size = pool_size
        x = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(stride_size, stride_size), padding=padding_type)(x)
        return x

    # Creating the inception STEM block        
    def inception_stem(self, x):
        net = self.conv2d_bn(x, filter_size=64, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.conv2d_bn(net, filter_size=64, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.maxpool_2d(net, pool_size=2, stride_size=2, padding_type='same')
        return net

    # Creating Inception v4- blockA (custom), blockA-reduction (custom)
    def inceptionv4_block_A(self, x):
        branch_0 = self.conv2d_bn(x, filter_size=64, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
    
        branch_1 = self.conv2d_bn(x, filter_size=32, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_1 = self.conv2d_bn(branch_1, filter_size=64, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
    
        branch_2 = self.conv2d_bn(x, filter_size=32, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_2 = self.conv2d_bn(branch_2, filter_size=64, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        branch_2 = self.conv2d_bn(branch_2, filter_size=64, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
    
        branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_3 = self.conv2d_bn(branch_3, filter_size=64, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
    
        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
        return x
    
    def inceptionv4_blockReduction_A(self, x):
        branch_0 = self.conv2d_bn(x, filter_size=256, kernel_size=3, padding_type='same', activation_type='LeakyRelu', strides=(2, 2))
    
        branch_1 = self.conv2d_bn(x, filter_size=128, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_1 = self.conv2d_bn(branch_1, filter_size=192, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_1 = self.conv2d_bn(branch_1, filter_size=256, kernel_size=1, padding_type='same', activation_type='LeakyRelu',
                                  strides=(2, 2))
    
        branch_2 = self.maxpool_2d(x, pool_size=3, stride_size=2, padding_type='same')
    
        x = concatenate([branch_0, branch_1, branch_2], axis=-1)
        return x

    # Creating Inception v4- blockB (custom), blockB-reduction (custom)
    def inceptionv4_block_B(self, x):
        branch_0 = self.conv2d_bn(x, filter_size=384, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
    
        branch_1 = self.conv2d_bn(x, filter_size=192, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_1 = self.conv2d_bn(branch_1, filter_size=224, kernel_size=(1, 7), padding_type='same', activation_type='LeakyRelu')
        branch_1 = self.conv2d_bn(branch_1, filter_size=256, kernel_size=(7, 1), padding_type='same', activation_type='LeakyRelu')
    
        branch_2 = self.conv2d_bn(x, filter_size=192, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_2 = self.conv2d_bn(branch_2, filter_size=192, kernel_size=(7, 1), padding_type='same', activation_type='LeakyRelu')
        branch_2 = self.conv2d_bn(branch_2, filter_size=224, kernel_size=(1, 7), padding_type='same', activation_type='LeakyRelu')
        branch_2 = self.conv2d_bn(branch_2, filter_size=224, kernel_size=(7, 1), padding_type='same', activation_type='LeakyRelu')
        branch_2 = self.conv2d_bn(branch_2, filter_size=256, kernel_size=(1, 7), padding_type='same', activation_type='LeakyRelu')
    
        branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_3 = self.conv2d_bn(branch_3, filter_size=128, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
    
        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
        return x
    
    def inceptionv4_blockReduction_B(self, x):
        branch_0 = self.conv2d_bn(x, filter_size=192, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_0 = self.conv2d_bn(x, filter_size=192, kernel_size=3, padding_type='same', activation_type='LeakyRelu', strides=(2, 2))
    
        branch_1 = self.conv2d_bn(x, filter_size=192, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_1 = self.conv2d_bn(branch_1, filter_size=192, kernel_size=(1, 7), padding_type='same', activation_type='LeakyRelu')
        branch_1 = self.conv2d_bn(branch_1, filter_size=256, kernel_size=(7, 1), padding_type='same', activation_type='LeakyRelu')
        branch_1 = self.conv2d_bn(branch_1, filter_size=256, kernel_size=3, padding_type='same', activation_type='LeakyRelu', strides=(2, 2))
    
        branch_2 = self.maxpool_2d(x, pool_size=3, stride_size=2, padding_type='same')
    
        x = concatenate([branch_0, branch_1, branch_2], axis=-1)
        return x

    # Creating Inception v4- blockC (custom)
    def inceptionv4_block_C(self, x):
        branch_0 = self.conv2d_bn(x, filter_size=256, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
    
        branch_1 = self.conv2d_bn(x, filter_size=384, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_10 = self.conv2d_bn(branch_1, filter_size=256, kernel_size=(1, 3), padding_type='same', activation_type='LeakyRelu')
        branch_11 = self.conv2d_bn(branch_1, filter_size=256, kernel_size=(3, 1), padding_type='same', activation_type='LeakyRelu')
        branch1 = concatenate([branch_10, branch_11], axis=-1)
    
        branch_2 = self.conv2d_bn(x, filter_size=384, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
        branch_2 = self.conv2d_bn(branch_2, filter_size=448, kernel_size=(3, 1), padding_type='same', activation_type='LeakyRelu')
        branch_2 = self.conv2d_bn(branch_2, filter_size=512, kernel_size=(1, 3), padding_type='same', activation_type='LeakyRelu')
        branch_2 = self.conv2d_bn(branch_2, filter_size=512, kernel_size=(1, 3), padding_type='same', activation_type='LeakyRelu')
        branch_20 = self.conv2d_bn(branch_2, filter_size=256, kernel_size=(1, 3), padding_type='same', activation_type='LeakyRelu')
        branch_21 = self.conv2d_bn(branch_2, filter_size=256, kernel_size=(3, 1), padding_type='same', activation_type='LeakyRelu')
        branch2 = concatenate([branch_20, branch_21], axis=-1)
    
        branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_3 = self.conv2d_bn(branch_3, filter_size=256, kernel_size=1, padding_type='same', activation_type='LeakyRelu')
    
        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
        return x

    def inceptionv4_custom(self, input_size, num_classes):
        input_layer = Input(input_size)
        # stem block of conv2d -> maxpool_2d layers
        net = self.inception_stem(input_layer)

        # call inception block A
        for idx in range(3):
            net = self.inceptionv4_block_A(net)

        # reduce inception block A
        net = self.inceptionv4_blockReduction_A(net)

        # call inception block B
        for idx in range(3):
            net = self.inceptionv4_block_B(net)

        # reduce inception block B
        net = self.inceptionv4_blockReduction_B(net)

        # call inception block C
        for idx in range(2):
            net = self.inceptionv4_block_C(net)

        # include top
        net = AveragePooling2D(pool_size=(8, 8), padding='valid')(net)
        net = Dropout(0.2)(net)
        net = Flatten()(net)
        net = Dense(units=num_classes, activation='softmax')(net)

        # final model
        model = Model(inputs=input_layer, outputs=net)
        return model


if __name__ == '__main__':
    input_size = (64, 64, 3)
    inceptionNet = InceptionV4()
    inception_model = inceptionNet.inceptionv4_custom(input_size)
    inception_model.summary()
