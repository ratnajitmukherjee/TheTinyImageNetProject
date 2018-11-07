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
 "
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: October 2018
"""

from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda, concatenate, AveragePooling2D, Add
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import VarianceScaling


class BuildNetworkModel:
    def __init__(self):
        print("\n Loading Network Model...")

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

    """
    Build a VGG like sequential network
    """
    def buildSequentialModel(self, inputsize, num_classes):
        input_layer = Input((64, 64, 3))
        # First block of conv2d -> Maxpool layers
        net = self.conv2d_bn(input_layer, filter_size=64, kernel_size=5, padding_type='same', activation_type='LeakyRelu')
        net = self.conv2d_bn(net, filter_size=64, kernel_size=5, padding_type='same', activation_type='LeakyRelu')
        net = self.maxpool_2d(net, pool_size=2, stride_size=2, padding_type='same')
        # second block of conv2d -> MaxPool layers
        net = self.conv2d_bn(net, filter_size=128, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.conv2d_bn(net, filter_size=128, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.maxpool_2d(net, pool_size=2, stride_size=2, padding_type='same')
        net = Dropout(0.1)(net)
        # Third block of conv2d -> MaxPool layers
        net = self.conv2d_bn(net, filter_size=256, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.conv2d_bn(net, filter_size=256, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.conv2d_bn(net, filter_size=256, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.maxpool_2d(net, pool_size=2, stride_size=2, padding_type='same')
        net = Dropout(0.15)(net)
        # Fourth block of conv2d -> MaxPool layers
        net = self.conv2d_bn(net, filter_size=512, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.conv2d_bn(net, filter_size=512, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.conv2d_bn(net, filter_size=512, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.maxpool_2d(net, pool_size=2, stride_size=2, padding_type='same')
        net = Dropout(0.2)(net)
        # Fifth block of conv2d -> MaxPool layers
        net = self.conv2d_bn(net, filter_size=512, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.conv2d_bn(net, filter_size=512, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        net = self.conv2d_bn(net, filter_size=512, kernel_size=3, padding_type='same', activation_type='LeakyRelu')
        # Flatten layer
        net = Flatten()(net)
        net = Dense(2048, activation='linear')(net)
        net = LeakyReLU(alpha=0.3)(net)
        net = Dense(2048, activation='linear')(net)
        net = LeakyReLU(alpha=0.3)(net)
        net = Dropout(0.5)(net)
        net = Dense(num_classes, activation='softmax')(net) 

        # Create the complete model
        model = Model(inputs=input_layer, outputs=net)    
        return model 

    """
    Build an Inception v4 type non-sequential network
    """       
    # def buildInceptionModel(self, inputsize, num_classes)


if __name__ == '__main__':
    print('STUD: NETWORK MODEL CLASS. BASELINE MODEL CREATED')
    # input and output layer parameters
    input_size = (64, 64, 3)
    num_classes = 200   

    # Calling the network building class
    buildNetwork = BuildNetworkModel()    
    seq_model = buildNetwork.buildSequentialModel(input_size, num_classes)    
    seq_model.summary()


