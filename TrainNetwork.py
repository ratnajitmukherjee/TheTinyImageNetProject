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
from ImagePreprocessing import ImageProcessor
from hdf5datasetgenerator import HDF5DatasetGenerator
from BuildTinyImageNetDataset import BuildTinyImageNetDataset
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
import os


class TrainTinyImageNet:
    def __init__(self, root_path):
        print("\n Training the TinyImageNet-200 dataset")
        self.root_path = root_path

    def model_plot_history(self, train):
        plt.plot(train.history['acc'], 'r+', linestyle='-', label='Training accuracy')
        plt.plot(train.history['loss'], 'b+', linestyle='-.', label='Training loss')

        plt.plot(train.history['val_acc'], 'rx', linestyle='-', label='Validation accuracy')
        plt.plot(train.history['val_loss'], 'bx', linestyle='-.', label='Validation loss')
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

        """
        Check whether the train, val and test files exist or not. 
        If they do skip building the dataset else build dataset and return mean
        Note: Later on replace the dataset mean with the actual mean returned during 
        dataset build from JSON dump.
        """

        if os.path.isfile(train_HDF5 and val_HDF5 and test_HDF5):
            print('\n The dataset already exists. Skipping dataset building process')
            rgb_mean = {'RMean': 123.68, 'GMean': 116.779, 'BMean': 103.939}
        else:
            rgb_mean = buildDataSet.buildDataSet()
        """
        data-augmentation and generating minibatches for training and validation
        """
        train_data_aug = ImageDataGenerator(rotation_range=25, zoom_range=0.5, width_shift_range=0.15,
                                            height_shift_range=0.15, shear_range=0.15, horizontal_flip=True,
                                            fill_mode='nearest')

        ip = ImageProcessor(width=input_size[0], height=input_size[1],
                            RMean=rgb_mean['RMean'], GMean=rgb_mean['GMean'], BMean=rgb_mean['BMean'],
                            dataFormat=None)

        trainGen = HDF5DatasetGenerator(dbPath=train_HDF5, batchSize=64, preprocessors=[ip],
                                        aug=train_data_aug, classes=num_classes)

        valGen = HDF5DatasetGenerator(dbPath=val_HDF5, batchSize=64, preprocessors=[ip], classes=num_classes)

        """
        load model and compile with custom optimization flags if required
        """
        buildNetwork = BuildNetworkModel()
        model = buildNetwork.buildSequentialModel(inputsize=input_size, num_classes=num_classes)
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

        """
        building checkpoint path and training
        """
        tiny_imagenet_checkpoints = os.path.join(root_path, 'TinyImageNet_checkpoint_{epoch:02d}-{val_acc:.2f}.hdf5')
        tiny_imagenet_callbacks = [EarlyStopping(monitor='val_loss', patience=15, mode='auto'),
                                   ModelCheckpoint(tiny_imagenet_checkpoints, monitor='val_acc', mode='auto', period=5)]

        tiny_imagenet_train = model.fit_generator(trainGen.generator(), trainGen.numImages//64, epochs=200,
                                                  verbose=True, validation_data=valGen.generator(),
                                                  validation_steps=valGen.numImages//64, max_queue_size=128,
                                                  callbacks=tiny_imagenet_callbacks)

        trainGen.close()
        valGen.close()

        model.save(filepath=os.path.join(self.root_path, 'TinyImageNetBaseline.hdf5'))
        print('\n TRAINING COMPLETE........ \n')

        # plot model loss and accuracy
        self.model_plot_history(train=tiny_imagenet_train)


if __name__ == '__main__':
    root_path = input('Please enter the root path: ')
    trainTinyImageNet = TrainTinyImageNet(root_path=root_path)
    input_size = (64, 64, 3)
    num_classes = 200
    trainTinyImageNet.train_tinyimagenet(input_size=input_size, num_classes=num_classes)
