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
from NetworkModel import BuildNetworkModel
from BuildTinyImageNetDataset import BuildTinyImageNetDataset
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt


class TrainTinyImageNet:
    def __init__(self, root_path):
        print("\n Training the TinyImageNet-200 dataset")
        self.root_path = root_path

    def model_plot_history(self, emotion_train):
        plt.plot(emotion_train.history['acc'], 'r+', linestyle='-', label='Training accuracy')
        plt.plot(emotion_train.history['loss'], 'b+', linestyle='-.', label='Training loss')

        plt.plot(emotion_train.history['val_acc'], 'rx', linestyle='-', label='Validation accuracy')
        plt.plot(emotion_train.history['val_loss'], 'bx', linestyle='-.', label='Validation loss')
        plt.minorticks_on()
        plt.ylabel("Model Training History")
        plt.xlabel("Epochs")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
        return

    def train_tinyimagenet(self, input_size, num_classes):
        buildDataSet = BuildTinyImageNetDataset(self.root_path)
        (train_HDF5, val_HDF5, test_HDF5) = buildDataSet.configDataSet()

        train_data_aug = ImageDataGenerator(rotation_range=25, zoom_range=0.5, width_shift_range=0.15,
                                            height_shift_range=0.15, shear_range=0.15, horizontal_flip=True,
                                            fill_mode='nearest')



if __name__ == '__main__':
    root_path = input('Please enter the root path: ')
    trainTinyImageNet = TrainTinyImageNet(root_path=root_path)