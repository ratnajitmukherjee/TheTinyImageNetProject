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
import json
import os

import keras.backend as K
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from BuildTinyImageNetDataset import BuildTinyImageNetDataset
from NetworkArchitectures.ResidualNetworkModel import ResNet
from Preprocessing.BasicPreprocessor import BasicPreprocessing
from Preprocessing.ImagetoArrayPreprocessor import ImagetoArrayPreprocessor
from Preprocessing.MeanPreprocessor import MeanPreprocessing
from hdf5io.hdf5datasetgenerator import HDF5DatasetGenerator


class TrainTinyImageNet:
    def __init__(self, root_path):
        print("\n Training the TinyImageNet-200 dataset")
        self.root_path = root_path

    def model_plot_history(self, train):
        plt.style.use('seaborn-darkgrid')
        plt.plot(train.history['acc'], 'r+', linestyle='-', label='Training accuracy')
        plt.plot(train.history['loss'], 'b+', linestyle='-.', label='Training loss')

        plt.plot(train.history['val_acc'], 'rx', linestyle='-', label='Validation accuracy')
        plt.plot(train.history['val_loss'], 'bx', linestyle='-.', label='Validation loss')
        plt.minorticks_on()
        plt.ylabel("Model Training History - Loss/Accuracy")
        plt.xlabel("Epochs")
        plt.legend(loc='upper right')
        plt.title('Residual Network Model Training History - TinyImageNet')
        plt.show()
        return

    def poly_decay(self, epoch):
        # Experimental polynomial decay function
        alpha = 1e-5 * (1 - (epoch / float(20))) ** 1.0
        return alpha

    def train_tinyimagenet(self, input_size, num_classes, pretrained_model, new_model_name, new_lr, num_epochs):
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
            print('Loading RGB mean from file {0}'.format(os.path.join(self.root_path, 'rgb_mean.txt')))
            fid = open(os.path.join(self.root_path, 'rgb_mean.txt'), 'r')
            rgb_mean = json.load(fid)
            fid.close()
        else:
            rgb_mean = buildDataSet.buildDataSet()
            fid = open(os.path.join(self.root_path, 'rgb_mean.txt'), 'w')
            fid.write(json.dumps(fid))
            print('RGB mean from the dataset written to file {0}'.format(os.path.join(self.root_path, 'rgb_mean.txt')))
            fid.close()
        """
        data-augmentation and generating minibatches for training and validation
        """
        train_data_aug = ImageDataGenerator(rotation_range=20, zoom_range=0.2, width_shift_range=0.1,
                                            height_shift_range=0.1, horizontal_flip=True,
                                            fill_mode='nearest')

        bp = BasicPreprocessing(input_size[0], input_size[1])
        mp = MeanPreprocessing(rgb_mean['RMean'], rgb_mean['GMean'], rgb_mean['BMean'])
        iap = ImagetoArrayPreprocessor()

        trainGen = HDF5DatasetGenerator(dbPath=train_HDF5, batchSize=64, preprocessors=[bp, mp, iap],
                                        aug=train_data_aug, classes=num_classes)

        valGen = HDF5DatasetGenerator(dbPath=val_HDF5, batchSize=64, preprocessors=[bp, mp, iap], classes=num_classes)

        """
        load model and compile with custom optimization flags if required
        """
        if pretrained_model is None:
            # Uncomment the next 2 lines to use the Sequential Network
            # buildNetwork = BuildNetworkModel()
            # model = buildNetwork.buildSequentialModel(inputsize=input_size, num_classes=num_classes)

            # Uncomment the next 2 lines to use the Inception V4 Network
            # inceptionNet = InceptionV4()
            # model = inceptionNet.inceptionv4_custom(input_size=input_size, num_classes=num_classes)

            """
            Residual Network (the stage list and filter list is only for ResNet. (Un)Comment whole block until 
            optimization if using Inception or Sequential Networks.
            """
            stage_list = (4, 5, 6)
            filter_list = (64, 128, 256, 512)
            resnet = ResNet()
            model = resnet.resnet_build(input_shape=input_size, num_classes=num_classes, filter_list=filter_list,
                                        stage_list=stage_list)

            # common optimizer (we can also use Adam/Nadam or SGD but SGD with poly decay attains better accuracy)
            sgd_opt = SGD(lr=self.init_lr, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd_opt, metrics=['accuracy'])

        else:
            pretr_model_path = os.path.join(self.root_path, pretrained_model)
            print('[INFO] Loading pretrained model: {0}'.format(pretr_model_path))
            model = load_model(pretr_model_path)

            if new_lr is None:
                old_learning_rate = K.get_value(model.optimizer.lr)
                new_lr = old_learning_rate // 10
                K.set_value(model.optimizer.lr, new_lr)
            else:
                old_learning_rate = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, new_lr)

            print('[INFO] Changing learning rate from {0} to {1}'.format(old_learning_rate, new_lr))

        # building checkpoint path and training
        tiny_imagenet_checkpoints = os.path.join(root_path, 'checkpoint_{epoch:02d}-{val_acc:.2f}.hdf5')

        tiny_imagenet_callbacks = [EarlyStopping(monitor='val_acc', patience=20, mode='auto'),
                                   ModelCheckpoint(tiny_imagenet_checkpoints, monitor='val_acc', mode='auto', period=5),
                                   LearningRateScheduler(self.poly_decay)]

        tiny_imagenet_train = model.fit_generator(trainGen.generator(), trainGen.numImages//64, epochs=num_epochs,
                                                  verbose=True, validation_data=valGen.generator(),
                                                  validation_steps=valGen.numImages//64, max_queue_size=128,
                                                  callbacks=tiny_imagenet_callbacks)

        trainGen.close()
        valGen.close()

        if new_model_name is None:
            new_model_name = 'TinyImageNetBaseline.hdf5'

        model.save(filepath=os.path.join(self.root_path, new_model_name))
        print('\n TRAINING COMPLETE........ \n')

        # plot model loss and accuracy
        self.model_plot_history(train=tiny_imagenet_train)


if __name__ == '__main__':
    root_path = input('Please enter the root path: ')
    trainTinyImageNet = TrainTinyImageNet(root_path=root_path)
    input_size = (64, 64, 3)
    num_classes = 200
    num_epochs = 120
    pretrained_model_name = 'TinyImageNet_ResNet_fn1.hdf5'
    new_model_name = 'TinyImageNet_ResNet_fn2.hdf5'
    new_lr = None
    trainTinyImageNet.train_tinyimagenet(input_size=input_size, num_classes=num_classes,
                                         pretrained_model=pretrained_model_name,
                                         new_model_name=new_model_name, new_lr=new_lr,
                                         num_epochs=num_epochs)